import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from chunker import chunk_text


def test_chunk_preserves_sentences_and_paragraphs():
    page = {
        'title': 'ParaTest',
        'content': (
            "This is sentence one. This is sentence two!\n\n"
            "A short paragraph.\n\n"
            "A long paragraph " + ("x" * 2000) + " end."
        )
    }
    chunks = chunk_text(page, chunk_size=800, overlap=100)
    # Should produce multiple chunks
    assert len(chunks) > 1
    # No chunk should be longer than chunk_size
    assert all(c['char_count'] <= 800 for c in chunks)
    # Ensure chunk texts contain whole sentences where possible
    # e.g., first sentence should appear intact in first chunk
    assert 'This is sentence one.' in chunks[0]['text']


def test_chunk_basic_repeat_chars():
    page = {'title': 'Test', 'content': 'A' * 2000}
    chunks = chunk_text(page, chunk_size=800, overlap=150)
    assert len(chunks) > 1
    assert all('text' in c and 'page_title' in c for c in chunks)
