from chunker import chunk_text
import json

# Use output from fetcher manual test
pages = [
    {
        "title": "TestPage",
        "content": "Dies ist ein Beispieltext. " * 100
    }
]

for page in pages:
    chunks = chunk_text(page, chunk_size=800, overlap=150)
    print(f"Page: {page['title']}, Chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks[:2]):  # Show first 2 chunks
        print(f"Chunk {i}: {json.dumps(chunk, ensure_ascii=False)[:200]}...")
    print("-" * 40)
