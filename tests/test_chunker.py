import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from chunker import chunk_text

def test_chunk_text_basic():
    page = {'title': 'Test', 'content': 'A' * 2000}
    chunks = chunk_text(page, chunk_size=800, overlap=150)
    assert len(chunks) > 1
    assert all('text' in c and 'page_title' in c for c in chunks)
