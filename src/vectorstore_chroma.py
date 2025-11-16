"""Chroma-backed vector store helper.

Provides a thin wrapper around `chromadb` to build, persist and query a
Chroma collection. This is optional; code importing this module should
handle ImportError when `chromadb` is not installed.
"""

from pathlib import Path
from typing import List, Tuple
import numpy as np


class ChromaVectorStore:
    def __init__(self, base_path: str, collection_name: str = 'mediawiki'):
        try:
            import chromadb
        except Exception as e:
            raise ImportError('chromadb not available') from e

        from chromadb.config import Settings
        self.chromadb = chromadb
        self.base_path = Path(base_path)
        self.persist_directory = str(self.base_path / 'chroma_db')
        self.collection_name = collection_name
        # create client with persistence
        self._client = chromadb.Client(Settings(persist_directory=self.persist_directory))
        # get or create collection
        try:
            self._collection = self._client.get_collection(self.collection_name)
        except Exception:
            self._collection = self._client.create_collection(self.collection_name)

    def build_index(self, embeddings: np.ndarray, chunks: List[dict]):
        # prepare ids, metadatas and documents
        ids = [str(i) for i in range(len(chunks))]
        metadatas = [{'page_title': c.get('page_title'), 'chunk_index': c.get('chunk_index')} for c in chunks]
        documents = [c.get('text', '') for c in chunks]
        embs = [list(map(float, e)) for e in np.array(embeddings, dtype='float32')]
        # add to collection
        self._collection.add(ids=ids, metadatas=metadatas, documents=documents, embeddings=embs)
        # persist
        try:
            self._client.persist()
        except Exception:
            pass

    def load(self):
        # client initialized in __init__, collection should already be available
        try:
            self._collection = self._client.get_collection(self.collection_name)
        except Exception:
            self._collection = self._client.create_collection(self.collection_name)

    def search(self, query_emb, top_k=5) -> Tuple[List[int], List[float]]:
        q = list(map(float, np.array(query_emb, dtype='float32')))
        res = self._collection.query(query_embeddings=[q], n_results=top_k)
        # chroma returns dict with 'ids' and 'distances' lists
        ids = [int(x) for x in res.get('ids', [[]])[0]]
        distances = res.get('distances', [[]])[0]
        # depending on chroma settings, distances may be similarity or distance
        return ids, distances
