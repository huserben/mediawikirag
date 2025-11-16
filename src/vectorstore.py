"""FAISS-backed vector store helper.

This module provides a thin wrapper around FAISS to build, save, load,
and query an index of embeddings stored as a NumPy array. If FAISS is
not available, the wrapper raises ImportError on construction so callers
can fall back to a numpy-based search.
"""

from pathlib import Path
import numpy as np


class FaissVectorStore:
    def __init__(self, base_path: str):
        try:
            import faiss
        except Exception as e:
            raise ImportError("faiss not available") from e

        self.faiss = faiss
        self.base_path = Path(base_path)
        self.index_path = self.base_path / 'faiss.index'
        self._index = None

    def build_index(self, embeddings: np.ndarray):
        """Build an in-memory FAISS index from `embeddings`.

        The index uses inner-product on L2-normalized vectors to implement
        cosine similarity.
        """
        if embeddings is None or len(embeddings) == 0:
            raise ValueError("No embeddings provided to build index")

        # Ensure float32
        embs = np.array(embeddings, dtype='float32')
        # Normalize for cosine similarity
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embs = embs / norms

        dim = embs.shape[1]
        index = self.faiss.IndexFlatIP(dim)
        index.add(embs)
        self._index = index

    def save(self):
        if self._index is None:
            raise RuntimeError('Index not built')
        self.faiss.write_index(self._index, str(self.index_path))

    def load(self):
        if not self.index_path.exists():
            raise FileNotFoundError('FAISS index not found')
        self._index = self.faiss.read_index(str(self.index_path))

    def search(self, query_emb, top_k=5):
        """Return (indices, scores) for the top_k nearest neighbors.

        `query_emb` should be a 1-D numpy array or list.
        """
        if self._index is None:
            raise RuntimeError('Index not built or loaded')

        import numpy as _np
        q = _np.array(query_emb, dtype='float32')
        # normalize
        norm = _np.linalg.norm(q)
        if norm != 0:
            q = q / norm
        q = q.reshape(1, -1)
        distances, indices = self._index.search(q, top_k)
        # FAISS IndexFlatIP returns inner products; treat as scores
        return indices[0].tolist(), distances[0].tolist()
