from fetcher import MediaWikiFetcher
from chunker import chunk_text
from embedder import Embedder
from storage import Storage
from retriever import cosine_similarity
import numpy as np
import yaml
import os

# Load config.yaml for cache path
with open(os.path.join(os.path.dirname(__file__), '../config.yaml'), 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
cache_dir = config['storage']['local_cache']

# Step 1: Fetch pages
api_url = config['wiki']['url'] + config['wiki']['api_endpoint']
fetcher = MediaWikiFetcher(api_url)
pages = fetcher.fetch_all_pages()

# Step 2: Chunk pages
all_chunks = []
for page in pages:
    chunks = chunk_text(page, chunk_size=config['chunking']['target_size'], overlap=config['chunking']['overlap'])
    all_chunks.extend(chunks)

# Step 3: Embed chunks
embedder = Embedder(config['models']['embedding'])
embeddings = embedder.embed_chunks(all_chunks)

# Step 4: Save to cache storage
storage = Storage(cache_dir)
storage.save_chunks(all_chunks)
storage.save_embeddings(embeddings)
loaded_chunks = storage.load_chunks()
loaded_embeddings = storage.load_embeddings()
print(f"Loaded {len(loaded_chunks)} chunks, Embeddings shape: {loaded_embeddings.shape}")

# Step 5: Query
query = "Wer ist Homer Simpson?"
query_emb = embedder.model.encode([query])[0]
scores = cosine_similarity(query_emb, loaded_embeddings)
top_k = config['retrieval']['top_k']
idxs = np.argsort(scores)[::-1][:top_k]
for i in idxs:
    print(f"Score: {scores[i]:.3f}, Chunk: {loaded_chunks[i]['text'][:100]}...")
