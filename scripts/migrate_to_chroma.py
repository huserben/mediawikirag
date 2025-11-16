"""Migration utility: convert existing chunks+embeddings into a Chroma DB.

Usage: python scripts/migrate_to_chroma.py --source D:/path/to/index

This will create or overwrite `source/chroma_db/` with a Chroma collection
containing the embeddings and chunk texts.
"""

import sys
import os
from pathlib import Path
import argparse
import json
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, help='Path to index folder (chunks.jsonl + embeddings.npy)')
    args = parser.parse_args()

    src = Path(args.source)
    chunks_path = src / 'chunks.jsonl'
    emb_path = src / 'embeddings.npy'
    if not chunks_path.exists() or not emb_path.exists():
        print('chunks.jsonl or embeddings.npy not found in source path')
        return

    chunks = [json.loads(line) for line in open(chunks_path, 'r', encoding='utf-8')]
    embs = np.load(str(emb_path))

    try:
        from vectorstore_chroma import ChromaVectorStore
    except Exception as e:
        print('chromadb not installed or vectorstore_chroma unavailable:', e)
        return

    cvs = ChromaVectorStore(src)
    cvs.build_index(embs, chunks)
    print('Chroma migration completed; persisted at', src / 'chroma_db')


if __name__ == '__main__':
    main()
