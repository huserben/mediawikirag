from retriever import cosine_similarity
from embedder import Embedder
from chunker import chunk_text
import numpy as np

# Sample page and query
page = {
    "title": "TestPage",
    "content": "Dies ist ein Beispieltext. " * 100
}
chunks = chunk_text(page, chunk_size=800, overlap=150)
texts = [chunk['text'] for chunk in chunks]
embedder = Embedder()
embeddings = embedder.model.encode(texts)

query = "Was ist ein Beispieltext?"
query_emb = embedder.model.encode([query])[0]
scores = cosine_similarity(query_emb, embeddings)
top_k = 2
idxs = np.argsort(scores)[::-1][:top_k]
for i in idxs:
    print(f"Score: {scores[i]:.3f}, Chunk: {chunks[i]['text'][:100]}...")
