# Search/similarity logic for Wiki RAG

import numpy as np


def cosine_similarity(query_emb, chunk_embs):
    query_emb = np.array(query_emb)
    chunk_embs = np.array(chunk_embs)
    dot = np.dot(chunk_embs, query_emb)
    norm_query = np.linalg.norm(query_emb)
    norm_chunks = np.linalg.norm(chunk_embs, axis=1)
    return dot / (norm_query * norm_chunks)
