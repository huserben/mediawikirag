"""
update_index.py - MediaWiki RAG Index Update Script

Handles:
- Lock acquisition and release
- Fetching pages from MediaWiki API
- Chunking and embedding
- Writing to staging/
- Atomic swap to current/
- Archiving old index
- Error handling and rollback
"""



import sys
import os
import logging
from pathlib import Path
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from config import load_config
from storage import Storage
from fetcher import MediaWikiFetcher
from chunker import chunk_text
from embedder import Embedder
import shutil
import datetime

logging.basicConfig(filename='update_index.log', level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def main():
    parser = argparse.ArgumentParser(description="Update MediaWiki RAG index.")
    parser.add_argument('--full-rebuild', action='store_true', help='Force full index rebuild')
    args = parser.parse_args()

    config = load_config()
    storage = Storage(config['storage']['network_drive'])
    lock_path = Path(config['storage']['network_drive']) / '.update.lock'
    staging_path = Path(config['storage']['network_drive']) / 'staging'
    current_path = Path(config['storage']['network_drive']) / 'current'
    archive_path = Path(config['storage']['network_drive']) / 'archive'
    import json
    try:
        try:
            storage.acquire_lock(lock_path)
            logging.info('Lock acquired.')
        except RuntimeError as le:
            print(f'[Lock Error] {le}')
            logging.error(f'Lock error: {le}')
            return

        wiki_url = config['wiki']['url'] + config['wiki']['api_endpoint']
        fetcher = MediaWikiFetcher(wiki_url)
        try:
            pages = fetcher.fetch_all_pages(max_retries=3, backoff=2, batch_size=50)
        except Exception as fetch_err:
            logging.error(f'Fetch error: {fetch_err}')
            print(f'[Fetch Error] {fetch_err}')
            storage.release_lock(lock_path)
            return
        logging.info(f'Fetched {len(pages)} pages.')

        state_path = current_path / 'wiki_state.json'
        if not args.full_rebuild and state_path.exists():
            with open(state_path, 'r', encoding='utf-8') as f:
                prev_state = json.load(f)
        else:
            prev_state = {"pages": {}}

        prev_pages = prev_state.get("pages", {})
        new_state = {"pages": {}}

        changed_pages = []
        for page in pages:
            title = page['title']
            revid = page.get('revid')
            timestamp = page.get('timestamp')
            new_state["pages"][title] = {"revid": revid, "timestamp": timestamp}
            prev = prev_pages.get(title)
            if args.full_rebuild or not prev or prev.get("revid") != revid:
                changed_pages.append(page)

        chunk_size = config['chunking']['target_size']
        overlap = config['chunking']['overlap']
        prev_chunks_path = current_path / 'chunks.jsonl'
        prev_chunks = []
        if not args.full_rebuild and prev_chunks_path.exists():
            with open(prev_chunks_path, 'r', encoding='utf-8') as f:
                prev_chunks = [json.loads(line) for line in f]

        all_chunks = []
        if not args.full_rebuild:
            unchanged_titles = set(prev_pages.keys()) & set(new_state["pages"].keys())
            for chunk in prev_chunks:
                title = chunk.get('page_title')
                if title in unchanged_titles and title not in [p['title'] for p in changed_pages]:
                    all_chunks.append(chunk)
        for page in changed_pages:
            chunks = chunk_text(page, chunk_size=chunk_size, overlap=overlap)
            all_chunks.extend(chunks)
        logging.info(f'Built chunk list with {len(all_chunks)} chunks.')

        embedder = Embedder(config['models']['embedding'])
        try:
            embeddings = embedder.embed_chunks(all_chunks)
        except Exception as embed_err:
            logging.error(f'Embedding error: {embed_err}')
            print(f'[Embedding Error] {embed_err}')
            storage.release_lock(lock_path)
            return
        logging.info(f'Generated embeddings for {len(all_chunks)} chunks.')

        try:
            storage.save_chunks(all_chunks, staging_path / 'chunks.jsonl')
            storage.save_embeddings(embeddings, staging_path / 'embeddings.npy')
            with open(staging_path / 'wiki_state.json', 'w', encoding='utf-8') as f:
                json.dump(new_state, f, ensure_ascii=False, indent=2)
            # Write metadata.json for versioning and stats
            metadata = {
                'version': datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
                'wiki_url': config['wiki'].get('url'),
                'total_pages': len(pages),
                'total_chunks': len(all_chunks),
                'model_name': config['models'].get('embedding'),
                'embedding_dim': int(embeddings.shape[1]) if hasattr(embeddings, 'shape') and len(embeddings.shape) > 1 else None,
                'chunk_size': chunk_size,
                'chunk_overlap': overlap
            }
            with open(staging_path / 'metadata.json', 'w', encoding='utf-8') as mf:
                json.dump(metadata, mf, ensure_ascii=False, indent=2)
            logging.info('Saved chunks, embeddings, and wiki_state to staging.')
        except Exception as write_err:
            logging.error(f'Write error: {write_err}')
            print(f'[Write Error] {write_err}')
            storage.release_lock(lock_path)
            return

        # 5. Validate integrity
        import hashlib
        def file_checksum(path):
            h = hashlib.sha256()
            try:
                with open(path, 'rb') as f:
                    while True:
                        chunk = f.read(8192)
                        if not chunk:
                            break
                        h.update(chunk)
                return h.hexdigest()
            except Exception as e:
                return None

        integrity_report = {}
        for fname in ['chunks.jsonl', 'embeddings.npy', 'wiki_state.json']:
            fpath = staging_path / fname
            exists = fpath.exists()
            checksum = file_checksum(fpath) if exists else None
            integrity_report[fname] = {'exists': exists, 'sha256': checksum}

        missing = [k for k, v in integrity_report.items() if not v['exists']]
        if missing:
            logging.error(f'Integrity check failed, missing files: {missing}')
            print(f'[Integrity Error] Missing files: {missing}')
            storage.release_lock(lock_path)
            return
        logging.info(f'Integrity check passed: {integrity_report}')

        try:
            if current_path.exists():
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
                archive_ver = archive_path / timestamp
                shutil.copytree(current_path, archive_ver)
                shutil.rmtree(current_path)
                logging.info(f'Archived previous index to {archive_ver}.')
                archives = sorted(archive_path.iterdir(), key=lambda p: p.name, reverse=True)
                for old in archives[2:]:
                    if old.is_dir():
                        shutil.rmtree(old)
                        logging.info(f'Removed old archive {old}.')
            shutil.copytree(staging_path, current_path)
            logging.info('Staging swapped to current.')
        except Exception as swap_err:
            logging.error(f'Swap error: {swap_err}')
            print(f'[Swap Error] {swap_err}')
            archives = sorted(archive_path.iterdir(), key=lambda p: p.name, reverse=True)
            if archives:
                latest = archives[0]
                if current_path.exists():
                    shutil.rmtree(current_path)
                shutil.copytree(latest, current_path)
                logging.info(f'Rollback: restored index from {latest}')
                print(f'[Rollback] Restored index from {latest}')
            storage.release_lock(lock_path)
            return

        storage.release_lock(lock_path)
        logging.info('Lock released.')
    except Exception as e:
        logging.error(f'Update failed: {e}')
        print(f'[Fatal Error] {e}')
        try:
            storage.release_lock(lock_path)
        except Exception:
            pass


if __name__ == '__main__':
    main()
