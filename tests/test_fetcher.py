import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from fetcher import MediaWikiFetcher

def test_fetcher_init():
    fetcher = MediaWikiFetcher('https://example.com/api.php')
    assert fetcher.api_url == 'https://example.com/api.php'
