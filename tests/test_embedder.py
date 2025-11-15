import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from embedder import Embedder

def test_embedder_init():
    embedder = Embedder()
    assert embedder.model is not None
