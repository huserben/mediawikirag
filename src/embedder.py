# Embedding generation for Wiki RAG

from sentence_transformers import SentenceTransformer

import numpy as np


class Embedder:

    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        self.model = SentenceTransformer(model_name)

    def embed_chunks(self, chunks):
        """
        Accepts a list of chunk dicts, returns numpy array of embeddings.
        """
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return np.array(embeddings, dtype=np.float32)
