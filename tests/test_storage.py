import tempfile
import os
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from storage import Storage

def test_storage_save_load_chunks():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = Storage(tmpdir)
        chunks = [{'id': 'c1', 'text': 'abc', 'page_title': 'Test', 'chunk_index': 0, 'char_count': 3}]
        storage.save_chunks(chunks)
        loaded = storage.load_chunks()
        assert loaded[0]['id'] == 'c1'
