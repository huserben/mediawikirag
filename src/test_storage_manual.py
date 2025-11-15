from storage import Storage
from chunker import chunk_text
import numpy as np
import tempfile
import os

# Sample page for storage test
page = {
    "title": "TestPage",
    "content": "Dies ist ein Beispieltext. " * 100
}
chunks = chunk_text(page, chunk_size=800, overlap=150)
embeddings = np.random.rand(len(chunks), 384).astype(np.float32)  # Fake embeddings

with tempfile.TemporaryDirectory() as tmpdir:
    storage = Storage(tmpdir)
    storage.save_chunks(chunks)
    storage.save_embeddings(embeddings)
    loaded_chunks = storage.load_chunks()
    loaded_embeddings = storage.load_embeddings()
    print(f"Loaded {len(loaded_chunks)} chunks, Embeddings shape: {loaded_embeddings.shape}")
    print(f"First loaded chunk: {loaded_chunks[0]}")
