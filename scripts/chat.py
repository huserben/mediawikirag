

import sys
import os
import logging
from pathlib import Path
import shutil
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from config import load_config
from storage import Storage
from embedder import Embedder
from cli import print_help

# Optional rich formatting
try:
    from rich.console import Console
    from rich.progress import Progress
    from rich.table import Table
    console = Console()
    USE_RICH = True
except ImportError:
    USE_RICH = False

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
        if USE_RICH:
            console.print(f"[yellow]{msg}[/yellow]")
        else:
            print(msg)
        for fname in INDEX_FILES:
            src = network_path / fname
            dst = cache_path / fname
            if src.exists():
                shutil.copy2(src, dst)
        msg = "Local cache updated."
        if USE_RICH:
            console.print(f"[green]{msg}[/green]")
        else:
            print(msg)
    else:
        msg = "Using cached index."
        if USE_RICH:
            console.print(f"[cyan]{msg}[/cyan]")
        else:
            print(msg)

def print_metadata(config):
    meta_path = Path(config['storage']['local_cache']) / METADATA_FILE
    meta = load_json(meta_path)
    if USE_RICH:
        table = Table(title="Index Metadata")
        table.add_column("Key")
        table.add_column("Value")
        for key in [
            'version', 'wiki_url', 'total_pages', 'total_chunks',
            'model_name', 'embedding_dim', 'chunk_size', 'chunk_overlap'
        ]:
            table.add_row(key.replace('_', ' ').title(), str(meta.get(key)))
        console.print(table)
    else:
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

def main():
    try:
        config = load_config()
        sync_local_cache(config)
        storage = Storage(config['storage']['local_cache'])
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
    if USE_RICH:
        console.print(f"[bold magenta]{ascii_art}[/bold magenta]")
        console.print("[bold green]Welcome to MediaWiki RAG Chat![/bold green]")
    else:
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
    if USE_RICH:
        console.print("[bold cyan]Available commands:[/bold cyan]")
        for cmd in commands:
            console.print(f"[cyan]{cmd}[/cyan]")
    else:
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
        if USE_RICH:
            console.print(f"[yellow]{msg}[/yellow]")
        else:
            print(msg)
        chunks = None
        embeddings = None
    else:
        # Progress indicator for index loading
        if USE_RICH:
            try:
                with Progress() as progress:
                    task = progress.add_task("[cyan]Loading index...", total=2)
                    storage.load_chunks()
                    progress.advance(task)
                    storage.load_embeddings()
                    progress.advance(task)
            except Exception as e:
                console.print(f"[red][Error] Failed to load index: {e}[/red]")
                chunks = None
                embeddings = None
        else:
            try:
                print("Loading index...")
                storage.load_chunks()
                storage.load_embeddings()
            except Exception as e:
                print(f"[Error] Failed to load index: {e}")
                chunks = None
                embeddings = None
    # ...existing code...
    embedder = Embedder(config['models']['embedding'])
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
                if USE_RICH:
                    console.print(f"[bold green]{msg}[/bold green]")
                else:
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
                if USE_RICH:
                    console.print(f"[yellow]{msg}[/yellow]")
                else:
                    print(msg)
                continue
            if cmd == '/update':
                msg = "Triggering full index rebuild..."
                if USE_RICH:
                    console.print(f"[yellow]{msg}[/yellow]")
                else:
                    print(msg)
                import subprocess
                try:
                    result = subprocess.run([
                        sys.executable,
                        'scripts/update_index.py',
                        '--full-rebuild'
                    ], capture_output=True, text=True)
                    if result.returncode == 0:
                        msg = "Index update completed. Use /refresh to reload."
                        if USE_RICH:
                            console.print(f"[green]{msg}[/green]")
                        else:
                            print(msg)
                    else:
                        msg = f"[Error] Update failed: {result.stderr.strip()}"
                        if USE_RICH:
                            console.print(f"[red]{msg}[/red]")
                        else:
                            print(msg)
                except Exception as e:
                    msg = f"[Error] Could not run update: {e}"
                    if USE_RICH:
                        console.print(f"[red]{msg}[/red]")
                    else:
                        print(msg)
                continue
            if cmd == '/refresh':
                if USE_RICH:
                    try:
                        with Progress() as progress:
                            task = progress.add_task("[cyan]Reloading index...", total=2)
                            storage.load_chunks()
                            progress.advance(task)
                            storage.load_embeddings()
                            progress.advance(task)
                            console.print("[green]Index reloaded from disk.[/green]")
                    except Exception as e:
                        console.print(f"[red][Error] Failed to reload index: {e}[/red]")
                else:
                    try:
                        print("Reloading index...")
                        storage.load_chunks()
                        storage.load_embeddings()
                        print("Index reloaded from disk.")
                    except Exception as e:
                        print(f"[Error] Failed to reload index: {e}")
                # Reload chunks and embeddings after refresh
                chunks = storage.chunks if hasattr(storage, 'chunks') else None
                embeddings = storage.embeddings if hasattr(storage, 'embeddings') else None
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
                if USE_RICH:
                    console.print(f"[red]{msg}[/red]")
                else:
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
                    if USE_RICH:
                        console.print(f"[yellow]{msg}[/yellow]")
                    else:
                        print(msg)
                    continue
                # Display results
                if USE_RICH:
                    table = Table(title="Search Results")
                    table.add_column("Score", justify="right")
                    table.add_column("Page")
                    table.add_column("Section")
                    table.add_column("Text", overflow="fold")
                    table.add_column("URL")
                    for chunk, score in results:
                        table.add_row(
                            f"{score:.3f}",
                            str(chunk.get('page_title', '')), 
                            str(chunk.get('section', '')), 
                            chunk.get('text', '')[:200] + ("..." if len(chunk.get('text', '')) > 200 else ""),
                            chunk.get('url', '')
                        )
                    console.print(table)
                else:
                    print("Results:")
                    for chunk, score in results:
                        print(f"Score: {score:.3f}")
                        print(f"Page: {chunk.get('page_title', '')}")
                        print(f"Section: {chunk.get('section', '')}")
                        print(f"Text: {chunk.get('text', '')[:200]}{'...' if len(chunk.get('text', '')) > 200 else ''}")
                        print(f"URL: {chunk.get('url', '')}")
                        print("-"*40)
            except Exception as e:
                msg = f"[Error] Query failed: {e}"
                if USE_RICH:
                    console.print(f"[red]{msg}[/red]")
                else:
                    print(msg)
        except Exception as e:
            print(f"[Error] Unexpected error: {e}")
