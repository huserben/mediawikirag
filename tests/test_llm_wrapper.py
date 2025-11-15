import sys
from pathlib import Path

import pytest


def test_gpt4all_adapter_with_mock(tmp_path, monkeypatch):
    # Create a fake model file
    model_file = tmp_path / "fake_model.bin"
    model_file.write_bytes(b"dummy")

    # Prepare a fake gpt4all module
    class FakeGPT4All:
        def __init__(self, model=None):
            self.model = model

        def generate(self, prompt, max_tokens=512, temperature=0.0):
            return "Antwort vom Mock-LLM"

    fake_mod = type(sys)('gpt4all')
    fake_mod.GPT4All = FakeGPT4All
    monkeypatch.setitem(sys.modules, 'gpt4all', fake_mod)

    from src.llm import get_llm

    cfg = {
        'models': {
            'llm': {
                'enabled': True,
                'provider': 'gpt4all',
                'model_path': str(model_file),
                'auto_download': False,
            }
        }
    }

    llm = get_llm(cfg)
    assert llm is not None
    out = llm.generate("Hallo")
    assert "Antwort vom Mock-LLM" in out
