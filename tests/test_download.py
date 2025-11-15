import sys
from pathlib import Path
import hashlib

import pytest


def test_ensure_model_download(tmp_path, monkeypatch):
    content = b"hello-model"
    sha = hashlib.sha256(content).hexdigest()

    class FakeResponse:
        def __init__(self, data):
            self.data = data

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size=8192):
            yield self.data

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_get(url, stream=True, timeout=30):
        return FakeResponse(content)

    monkeypatch.setattr('requests.get', fake_get)

    from src.download import ensure_model

    dest = tmp_path / "model.bin"
    path = ensure_model(str(dest), "http://example.com/model.bin", sha256=sha)
    assert Path(path).exists()
    # verify checksum
    with open(path, 'rb') as f:
        data = f.read()
    assert hashlib.sha256(data).hexdigest() == sha
