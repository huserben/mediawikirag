import sys
import os

# Ensure the project root is on sys.path when running tests directly (helps
# when running a single test). pytest normally handles this when running the
# entire suite.
sys.path.insert(0, os.path.abspath(os.curdir))

# pytest fixtures are provided by the test runner (monkeypatch, tmp_path)


def test_gpt4all_adapter_with_mock(tmp_path, monkeypatch):
    # Create a fake model file
    model_file = tmp_path / "fake_model.bin"
    model_file.write_bytes(b"dummy")

    # Prepare a fake gpt4all module
    class FakeGPT4All:
        def __init__(self, *a, **kw):
            # accept arbitrary args/kwargs to mimic upstream GPT4All
            self._init_args = (a, kw)

        def generate(self, prompt, *a, **kw):
            return "Antwort vom Mock-LLM"

    fake_mod = type(sys)('gpt4all')
    fake_mod.GPT4All = FakeGPT4All
    monkeypatch.setitem(sys.modules, 'gpt4all', fake_mod)

    # src.llm will import gpt4all in _load, the test above sets sys.modules
    # for the fake gpt4all already.

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


def test_ollama_adapter_with_mock(monkeypatch):
    # Fake responses for requests.get and requests.post
    class FakeResponse:
        def __init__(self, data=None, status_code=200):
            self._data = data or {}
            self.status_code = status_code

        def raise_for_status(self):
            if self.status_code >= 400:
                raise ValueError("error")

        def json(self):
            return self._data

    def fake_get(url, timeout=None):
        # Accept either /api/tags, /tags or /api/models paths
        if url.endswith("/api/tags"):
            return FakeResponse({"tags": ["gemma2:2b"]}, 200)
        if url.endswith("/tags"):
            return FakeResponse({"tags": ["gemma2:2b"]}, 200)
        return FakeResponse({"models": ["my-model"]}, 200)

    def fake_post(url, json, timeout=None):
        return FakeResponse({"text": "Antwort von Ollama"}, 200)

    # Patch the global requests library before importing src.llm so
    # that the module uses the mocked functions at import time.
    monkeypatch.setattr("requests.get", fake_get)
    monkeypatch.setattr("requests.post", fake_post)

    from src.llm import get_llm

    cfg = {
        'models': {
            'llm': {
                'enabled': True,
                'provider': 'ollama',
                'model': 'my-model',
                'base_url': 'http://localhost:11434',
            }
        }
    }

    llm = get_llm(cfg)
    assert llm is not None
    out = llm.generate("Hallo")
    assert "Antwort von Ollama" in out

    def test_ollama_adapter_ignores_model_key(monkeypatch):
        class FakeResponse:
            def __init__(self, data=None, status_code=200):
                self._data = data or {}
                self.status_code = status_code

            def raise_for_status(self):
                if self.status_code >= 400:
                    raise ValueError("error")

            def json(self):
                return self._data

        def fake_get(url, timeout=None):
            return FakeResponse({"tags": ["gemma2:2b"]}, 200)

        def fake_post(url, json, timeout=None):
            return FakeResponse({"model": "gemma2:2b"}, 200)

        monkeypatch.setattr("requests.get", fake_get)
        monkeypatch.setattr("requests.post", fake_post)

        import sys
        import os
        sys.path.insert(0, os.path.abspath(os.curdir))
        from src.llm import get_llm, LLMError

        cfg = {
            'models': {'llm': {'enabled': True, 'provider': 'ollama', 'model': 'gemma'}}
        }

        llm = get_llm(cfg)
        assert llm is not None
        import pytest
        with pytest.raises(LLMError):
            _ = llm.generate("Should ignore model key")


    def test_ollama_adapter_prefers_text_over_model(monkeypatch):
        class FakeResponse:
            def __init__(self, data=None, status_code=200):
                self._data = data or {}
                self.status_code = status_code

            def raise_for_status(self):
                if self.status_code >= 400:
                    raise ValueError("error")

            def json(self):
                return self._data

        def fake_get(url, timeout=None):
            return FakeResponse({"tags": ["gemma2:2b"]}, 200)

        def fake_post(url, json, timeout=None):
            return FakeResponse({"model": "gemma2:2b", "text": "This is the proper text"}, 200)

        monkeypatch.setattr("requests.get", fake_get)
        monkeypatch.setattr("requests.post", fake_post)

        import sys
        import os
        sys.path.insert(0, os.path.abspath(os.curdir))
        from src.llm import get_llm

        cfg = {
            'models': {'llm': {'enabled': True, 'provider': 'ollama', 'model': 'gemma'}}
        }

        llm = get_llm(cfg)
        assert llm is not None
        out = llm.generate("Which text should we choose?")
        assert "proper text" in out


    def test_ollama_auto_fallback_to_messages(monkeypatch):
        """If prompt returns no text, adapter should fallback to messages when request_format is 'auto'."""

        class FakeResponse:
            def __init__(self, data=None, status_code=200, text=None):
                self._data = data or {}
                self._status_code = status_code
                self._text = text or ""

            def raise_for_status(self):
                if self._status_code >= 400:
                    raise ValueError("error")

            def json(self):
                return self._data

            @property
            def text(self):
                return self._text

        def fake_get(url, timeout=None):
            return FakeResponse({"tags": ["gemma2:2b"]}, 200)

        def fake_post(url, json, timeout=None):
            # If prompt format is used first, return metadata-only (no text), next messages returns real text
            if "prompt" in json:
                return FakeResponse({"model": "gemma2:2b", "response": ""}, 200)
            if "messages" in json:
                return FakeResponse({"text": "Answer from messages"}, 200)

            return FakeResponse({})

        monkeypatch.setattr("requests.get", fake_get)
        monkeypatch.setattr("requests.post", fake_post)

        import sys
        import os
        sys.path.insert(0, os.path.abspath(os.curdir))
        from src.llm import get_llm

        cfg = {
            'models': {'llm': {'enabled': True, 'provider': 'ollama', 'model': 'gemma', 'request_format': 'auto'}}
        }

        llm = get_llm(cfg)
        assert llm is not None
        out = llm.generate("Why is the sky blue?")
        assert "Answer from messages" in out


    def test_ollama_respects_messages_format(monkeypatch):
        """If request_format is explicitly 'messages', use messages only."""

        class FakeResponse:
            def __init__(self, data=None, status_code=200):
                self._data = data or {}
                self.status_code = status_code

            def raise_for_status(self):
                if self.status_code >= 400:
                    raise ValueError("error")

            def json(self):
                return self._data

        def fake_get(url, timeout=None):
            return FakeResponse({"tags": ["gemma2:2b"]}, 200)

        def fake_post(url, json, timeout=None):
            assert "messages" in json and "prompt" not in json
            return FakeResponse({"text": "messages-only answer"}, 200)

        monkeypatch.setattr("requests.get", fake_get)
        monkeypatch.setattr("requests.post", fake_post)

        import sys
        import os
        sys.path.insert(0, os.path.abspath(os.curdir))
        from src.llm import get_llm

        cfg = {
            'models': {'llm': {'enabled': True, 'provider': 'ollama', 'model': 'gemma', 'request_format': 'messages'}}
        }

        llm = get_llm(cfg)
        assert llm is not None
        out = llm.generate("Tell me something")
        assert "messages-only answer" in out


def test_ollama_adapter_with_ndjson_stream(monkeypatch):
    class FakeResponse:
        def __init__(self, text, status_code=200):
            self._text = text
            self.status_code = status_code

        def raise_for_status(self):
            if self.status_code >= 400:
                raise ValueError("error")

        def json(self):
            raise ValueError("Invalid JSON for streaming")

        @property
        def text(self):
            return self._text

    # Simulate SSE-like streaming with multiple data lines
    ndjson = 'data: {"text": "partial"}\n\n' + 'data: {"text": "final"}'

    def fake_get(url, timeout=None):
        return FakeResponse('{"tags": ["gemma2:2b"]}', 200)

    def fake_post(url, json, timeout=None):
        return FakeResponse(ndjson, 200)

    monkeypatch.setattr("requests.get", fake_get)
    monkeypatch.setattr("requests.post", fake_post)

    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.curdir))
    from src.llm import get_llm

    cfg = {
        'models': {'llm': {'enabled': True, 'provider': 'ollama', 'model': 'gemma'}}
    }

    llm = get_llm(cfg)
    assert llm is not None
    out = llm.generate("Hello")
    assert "final" in out


def test_ollama_adapter_with_response_key(monkeypatch):
    class FakeResponse:
        def __init__(self, data, status_code=200):
            self._data = data
            self.status_code = status_code

        def raise_for_status(self):
            if self.status_code >= 400:
                raise ValueError("error")

        def json(self):
            return self._data

    def fake_get(url, timeout=None):
        return FakeResponse({"tags": ["gemma2:2b"]}, 200)

    def fake_post(url, json, timeout=None):
        return FakeResponse({"response": "Homer Simpson ist 39 jahre alt"}, 200)

    monkeypatch.setattr("requests.get", fake_get)
    monkeypatch.setattr("requests.post", fake_post)

    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.curdir))
    from src.llm import get_llm

    cfg = {
        'models': {'llm': {'enabled': True, 'provider': 'ollama', 'model': 'gemma'}}
    }

    llm = get_llm(cfg)
    assert llm is not None
    out = llm.generate("Wie alt ist Homer Simpson?")
    assert "Homer" in out


def test_ollama_adapter_empty_response_triggers_error(monkeypatch):
    class FakeResponse:
        def __init__(self, data, status_code=200):
            self._data = data
            self.status_code = status_code

        def raise_for_status(self):
            if self.status_code >= 400:
                raise ValueError("error")

        def json(self):
            return self._data

    def fake_get(url, timeout=None):
        return FakeResponse({"tags": ["gemma2:2b"]}, 200)

    def fake_post(url, json, timeout=None):
        return FakeResponse({"response": ""}, 200)

    monkeypatch.setattr("requests.get", fake_get)
    monkeypatch.setattr("requests.post", fake_post)

    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.curdir))
    from src.llm import get_llm, LLMError

    cfg = {
        'models': {'llm': {'enabled': True, 'provider': 'ollama', 'model': 'gemma'}}
    }

    llm = get_llm(cfg)
    assert llm is not None
    import pytest
    with pytest.raises(LLMError):
        _ = llm.generate("empty response test")
