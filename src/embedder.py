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

    def embed(self, texts):
        """
        Embed a single string or a list of strings. Returns a numpy array
        (n_texts, dim) or a single vector if input was a single string.
        """
        if isinstance(texts, str):
            single = True
            items = [texts]
        else:
            single = False
            items = list(texts)
        embeddings = self.model.encode(items, show_progress_bar=False)
        arr = np.array(embeddings, dtype=np.float32)
        return arr[0] if single else arr
