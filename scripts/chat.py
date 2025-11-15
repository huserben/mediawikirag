

import sys
import os
import logging
from pathlib import Path
import shutil
import json

import textwrap
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from config import load_config
from storage import Storage
from embedder import Embedder
from cli import print_help
from llm import get_llm, LLMError
from prompts import load_prompt


logging.basicConfig(
    filename='chat.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

METADATA_FILE = 'metadata.json'
INDEX_FILES = ['chunks.jsonl', 'embeddings.npy', METADATA_FILE, 'wiki_state.json']


def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def sync_local_cache(config):
    network_path = Path(config['storage']['network_drive']) / 'current'
    cache_path = Path(config['storage']['local_cache'])
    cache_path.mkdir(parents=True, exist_ok=True)
    net_meta = network_path / METADATA_FILE
    cache_meta = cache_path / METADATA_FILE
    net_info = load_json(net_meta)
    cache_info = load_json(cache_meta)
    if net_info.get('version') != cache_info.get('version'):
        msg = "New index version detected, syncing local cache..."
        print(msg)
        for fname in INDEX_FILES:
            src = network_path / fname
            dst = cache_path / fname
            if src.exists():
                shutil.copy2(src, dst)
        print("Local cache updated.")
    else:
        print("Using cached index.")


def print_metadata(config):
    meta_path = Path(config['storage']['local_cache']) / METADATA_FILE
    meta = load_json(meta_path)
    print("Index Metadata:")
    for key in [
        'version', 'wiki_url', 'total_pages', 'total_chunks',
        'model_name', 'embedding_dim', 'chunk_size', 'chunk_overlap'
    ]:
        print(f"  {key.replace('_', ' ').title()}: {meta.get(key)}")


def print_config(retrieval):
    print("Current search parameters:")
    print(f"  top_k: {retrieval.get('top_k', 5)}")
    print(f"  similarity_threshold: {retrieval.get('similarity_threshold', 0.3)}")
    print("Type 'top_k <value>' or 'threshold <value>' to change, or just press Enter to keep.")

def synthesize_answer(results):
    """
    Given `results` as a list of (chunk, score) tuples, synthesize a short,
    human-readable extractive answer and a list of sources. Returns
    a tuple `(answer_text, sources_list)`.
    """
    # Prepare compact context entries
    entries = []
    for chunk, score in results:
        page = chunk.get('page_title') or chunk.get('page', 'Unknown')
        section = chunk.get('section') or ''
        text = chunk.get('text', '')
        url = chunk.get('url', '')
        # truncate to keep prompts reasonably small
        snippet = textwrap.shorten(text.replace('\n', ' '), width=800, placeholder='...')
        entries.append({'page': page, 'section': section, 'text': snippet, 'url': url, 'score': score})

    # Offline extractive summarizer: combine top snippets and produce a
    # short human-readable answer with concise source citations.
    if not entries:
        return ("Entschuldigung, ich konnte keine Informationen finden.", [])

    # Combine the top 2 snippets to increase coverage
    combined = ' '.join(e['text'] for e in entries[:2])
    # Simple sentence split (works adequately for short snippets)
    import re
    sentences = re.split(r'(?<=[.!?])\s+', combined)
    answer_sentences = [s.strip() for s in sentences if s.strip()][:2]
    answer = ' '.join(answer_sentences)
    if not answer:
        answer = entries[0]['text'][:400].strip()
    if answer and not answer.endswith(('.', '!', '?')):
        answer = answer + '.'

    # Build concise source list (page, url)
    sources = [(e['page'], e['url']) for e in entries]
    source_names = ', '.join([s[0] for s in sources[:3]])
    answer_with_source = f"{answer}\n\nQuelle: {source_names}"
    return answer_with_source, sources


def main():
    try:
        config = load_config()
        sync_local_cache(config)
        storage = Storage(config['storage']['local_cache'])
        # Instantiate LLM and require it to be available when enabled
        llm = get_llm(config)
    except Exception as e:
        print(f"[Fatal Error] Startup failed: {e}")
        return

    # ASCII art and greeting
    ascii_art = r"""
 __  __          _      _       _ _    _ _    _ _    _
|  \/  | ___  __| | ___| | __ _| | | _(_) | _(_) | _(_)
| |\/| |/ _ \/ _` |/ _ \ |/ _` | | |/ / | |/ / | |/ / |
| |  | |  __/ (_| |  __/ | (_| | |   <| |   <| |   <| |
|_|  |_|\___|\__,_|\___|_|\__,_|_|_|\_\_|_|\_\_|_|\_\_|
"""
    print(ascii_art)
    print("Welcome to MediaWiki RAG Chat!")

    # Command summary
    commands = [
        "/help   - Show help",
        "/info   - Show index metadata",
        "/config - View/change search parameters",
        "/refresh- Reload index from disk",
        "/update - Trigger full index rebuild",
        "/quit   - Exit chat"
    ]
    print("Available commands:")
    for cmd in commands:
        print(cmd)

    # Check for index/cache files before loading
    cache_path = Path(config['storage']['local_cache'])
    missing_files = [fname for fname in INDEX_FILES if not (cache_path / fname).exists()]
    if missing_files:
        msg = (
            "No index found. Please run /update to build the index before "
            "searching."
        )
        print(msg)
        chunks = None
        embeddings = None
    else:
        # Progress indicator for index loading
        try:
            print("Loading index...")
            chunks = storage.load_chunks()
            embeddings = storage.load_embeddings()
        except Exception as e:
            print(f"[Error] Failed to load index: {e}")
            chunks = None
            embeddings = None
    # ...existing code...
    embedder = Embedder(config['models']['embedding'])
    # LLM is required (if enabled) and was initialized at startup
    chunks = storage.chunks if hasattr(storage, 'chunks') else None
    embeddings = storage.embeddings if hasattr(storage, 'embeddings') else None
    retrieval = config.get('retrieval', {})
    top_k = retrieval.get('top_k', 5)
    similarity_threshold = retrieval.get('similarity_threshold', 0.3)

    while True:
        try:
            query = input('> ')
            logging.info(f"User input: {query}")
            cmd = query.strip()
            if cmd in ['/quit', '/exit']:
                msg = "Goodbye!"
                print(msg)
                break
            elif cmd == '/help':
                print_help()
                continue
            elif cmd == '/info':
                print_metadata(config)
                continue
            if not cmd:
                msg = "Please enter a question or command."
                print(msg)
                continue
            if cmd == '/update':
                msg = "Triggering full index rebuild..."
                print(msg)
                import subprocess
                try:
                    result = subprocess.run([
                        sys.executable,
                        'scripts/update_index.py',
                        '--full-rebuild'
                    ], capture_output=True, text=True)
                    if result.returncode == 0:
                        msg = "Index update completed. Syncing local cache and reloading recommended."
                        print(msg)
                        try:
                            sync_local_cache(config)
                        except Exception as e:
                            print(f"[Warning] Sync after update failed: {e}")
                    else:
                        msg = f"[Error] Update failed: {result.stderr.strip()}"
                        print(msg)
                except Exception as e:
                    msg = f"[Error] Could not run update: {e}"
                    print(msg)
                continue
            if cmd == '/refresh':
                try:
                    print("Refreshing local cache from network and reloading index...")
                    try:
                        sync_local_cache(config)
                    except Exception as e:
                        print(f"[Warning] Sync failed: {e}")
                    chunks = storage.load_chunks()
                    embeddings = storage.load_embeddings()
                    print("Index reloaded from disk.")
                except Exception as e:
                    print(f"[Error] Failed to reload index: {e}")
                # Reload chunks and embeddings after refresh
                # variables already updated above
                continue
            if cmd == '/config':
                retrieval = config.get('retrieval', {})
                print_config(retrieval)
                subcmd = input('config> ').strip()
                if subcmd.startswith('top_k '):
                    try:
                        new_k = int(subcmd.split()[1])
                        config['retrieval']['top_k'] = new_k
                        top_k = new_k
                        print(f"top_k set to {new_k}")
                    except Exception:
                        print("Invalid value for top_k.")
                elif subcmd.startswith('threshold '):
                    try:
                        new_thresh = float(subcmd.split()[1])
                        config['retrieval']['similarity_threshold'] = new_thresh
                        similarity_threshold = new_thresh
                        print(f"similarity_threshold set to {new_thresh}")
                    except Exception:
                        print("Invalid value for threshold.")
                else:
                    print("No changes made.")
                continue

            # Query logic: embed, search, display results
            if chunks is None or embeddings is None:
                msg = "No index loaded. Please run /update and /refresh."
                print(msg)
                continue
            try:
                # Embed user query
                query_vec = embedder.embed([cmd])[0]
                import numpy as np
                emb_matrix = np.array(embeddings)
                # Compute cosine similarity
                norm_emb = emb_matrix / np.linalg.norm(emb_matrix, axis=1, keepdims=True)
                norm_query = query_vec / np.linalg.norm(query_vec)
                scores = np.dot(norm_emb, norm_query)
                # Get top-k results above threshold
                top_indices = np.argsort(scores)[::-1][:top_k]
                results = [
                    (chunks[i], float(scores[i]))
                    for i in top_indices if scores[i] >= similarity_threshold
                ]
                if not results:
                    msg = "No relevant results found. Try rephrasing your question."
                    print(msg)
                    continue
                # If an LLM is configured, build context and ask it to generate
                answer_text = None
                sources = [(r[0].get('page_title') or r[0].get('page', 'Unknown'), r[0].get('url', '')) for r in results]
                if llm is not None:
                    try:
                        # Build a compact context from top results
                        def build_context(results_list, max_chars=3000):
                            parts = []
                            chars = 0
                            for chunk, score in results_list:
                                text = chunk.get('text', '')
                                meta = chunk.get('page_title') or chunk.get('page', '')
                                part = f"Quelle: {meta}\n{text}\n"
                                if chars + len(part) > max_chars:
                                    break
                                parts.append(part)
                                chars += len(part)
                            return "\n---\n".join(parts)

                        context_text = build_context(results)
                        prompt_path = config.get('models', {}).get('llm', {}).get('prompt_template_path', 'prompts/german_default.txt')
                        try:
                            template = load_prompt(prompt_path)
                        except Exception:
                            template = None

                        if template:
                            prompt = template.format(context=context_text, question=cmd)
                        else:
                            prompt = f"Kontext:\n{context_text}\n\nFrage: {cmd}\n\nAntwort:" 

                        answer_text = llm.generate(prompt, max_tokens=config.get('models', {}).get('llm', {}).get('max_tokens', 512), temperature=config.get('models', {}).get('llm', {}).get('temperature', 0.0))
                    except LLMError as e:
                        print(f"[Warning] LLM generation failed: {e}. Falling back to extractive answer.")
                        answer_text = None
                    except Exception as e:
                        print(f"[Warning] Unexpected LLM error: {e}. Falling back to extractive answer.")
                        answer_text = None

                # Synthesize a short human-readable answer from the retrieved contexts (fallback)
                if not answer_text:
                    answer_text, _ = synthesize_answer(results)
                print("\nAntwort:")
                print(answer_text)
                if sources:
                    print("\nQuellen:")
                    for p, url in sources[:5]:
                        if url:
                            print(f" - {p}: {url}")
                        else:
                            print(f" - {p}")
                # continue to prompt for next query
                continue
            except Exception as e:
                msg = f"[Error] Query failed: {e}"
                print(msg)
        except Exception as e:
            print(f"[Error] Unexpected error: {e}")


if __name__ == '__main__':
    main()
