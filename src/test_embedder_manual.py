from embedder import Embedder
from chunker import chunk_text

# Sample page for embedding test
page = {
    "title": "TestPage",
    "content": "Dies ist ein Beispieltext. " * 100
}
chunks = chunk_text(page, chunk_size=800, overlap=150)
texts = [chunk['text'] for chunk in chunks[:2]]  # Use first 2 chunks

embedder = Embedder()
embeddings = embedder.model.encode(texts)
print(f"Embeddings shape: {embeddings.shape}")
print(f"First embedding: {embeddings[0][:10]}")
