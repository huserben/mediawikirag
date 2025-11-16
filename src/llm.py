"""
Lightweight LLM adapter layer.
Provides a simple interface and a GPT4All adapter with optional
auto-download support via `src.download.ensure_model`.
"""

from typing import Optional, List
from dataclasses import dataclass


class LLMError(Exception):
    pass


class LLM:
    """Abstract LLM interface."""

    def generate(
        self, prompt: str, max_tokens: int = 512, temperature: float = 0.0
    ) -> str:
        raise NotImplementedError()


@dataclass
class GPT4AllAdapter(LLM):
    model_path: str
    n_ctx: int = 2048
    device: str = "cpu"
    max_tokens: int = 512
    temperature: float = 0.0
    auto_download: bool = False
    model_url: Optional[str] = None
    model_sha256: Optional[str] = None
    prompt_for_download: bool = True

    def __post_init__(self):
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            # Lazily import GPT4All to avoid import-time dependency
            # errors in test environments.
            from gpt4all import GPT4All

            self._model = GPT4All(
                model_name=self.model_path,
                allow_download=self.auto_download,
                device=self.device,
            )
        except Exception as e:
            raise LLMError(f"Failed to load GPT4All model: {e}")

    def generate(
        self, prompt: str, max_tokens: int = None, temperature: float = None
    ) -> str:
        self._load()

        if max_tokens is None:
            max_tokens = self.max_tokens
        if temperature is None:
            temperature = self.temperature

        try:
            # GPT4All's generate method returns a string directly
            output = self._model.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temp=temperature,  # Fixed: parameter name is 'temp' not 'temperature'
            )
            return str(output)
        except Exception as e:
            raise LLMError(f"LLM generation failed: {e}")

    def generate_from_chunks(
        self,
        chunks: List[str],
        prompt_template: str,
        question: str,
        max_tokens: int = None,
        temperature: float = None,
    ) -> str:
        """Compatibility wrapper: render the prompt template with `{context}` and
        `{question}` then call the normal `generate` method.
        """
        if max_tokens is None:
            max_tokens = self.max_tokens
        if temperature is None:
            temperature = self.temperature

        context = "\n\n".join(chunks or [])
        try:
            prompt = prompt_template.format(context=context, question=question)
        except Exception:
            prompt = f"{context}\n\nQuestion: {question}"

        return self.generate(prompt, max_tokens=max_tokens, temperature=temperature)


@dataclass
class OllamaAdapter(LLM):
    """Adapter that uses the official `ollama` Python client.

    This adapter exposes a structured, synchronous API `generate_from_chunks`
    which renders a prompt template with `{context}` and `{question}` and
    calls the local Ollama server via `ollama.Client`.
    """

    model_name: str
    base_url: str = "http://127.0.0.1:11434"
    max_tokens: int = 512
    temperature: float = 0.0
    timeout: int = 30

    def __post_init__(self):
        self._available = False
        self._client = None

    def _load(self):
        # Lazily import and instantiate the Ollama client. On failure,
        # raise LLMError so callers know the adapter is not usable.
        if self._client is not None:
            return

        try:
            from ollama import Client

            self._client = Client(host=self.base_url)
        except Exception as e:
            raise LLMError(f"Failed to load ollama client: {e}")

        # Try to verify model availability if the client exposes a models()
        # or show() method; if verification fails, surface an error.
        try:
            models_fn = getattr(self._client, "models", None)
            if callable(models_fn):
                models = models_fn()
                present = False
                try:
                    if isinstance(models, (list, tuple)):
                        for m in models:
                            if isinstance(m, str) and m == self.model_name:
                                present = True
                                break
                            if isinstance(m, dict) and m.get("name") == self.model_name:
                                present = True
                                break
                except Exception:
                    pass
                if not present:
                    # Try show() if available
                    show_fn = getattr(self._client, "show", None)
                    if callable(show_fn):
                        show_fn(self.model_name)
                        present = True
                # If we couldn't verify presence, still allow client to be used
            self._available = True
        except Exception as e:
            raise LLMError(f"Failed to verify Ollama model availability: {e}")

    def generate_from_chunks(
        self,
        chunks: List[str],
        prompt_template: str,
        question: str,
        max_tokens: int = None,
        temperature: float = None,
    ) -> str:
        """Render the template with the provided chunks and question, call
        the Ollama client synchronously, and return the model's text output.

        The template should include `{context}` and `{question}` placeholders.
        """
        self._load()

        if max_tokens is None:
            max_tokens = self.max_tokens
        if temperature is None:
            temperature = self.temperature

        # Assemble context and render template
        context = "\n\n".join(chunks or [])
        try:
            prompt = prompt_template.format(context=context, question=question)
        except Exception:
            # Fallback: simple concatenation
            prompt = f"{context}\n\nQuestion: {question}"

        try:
            # Prefer a generate() method if present, else try chat()
            if hasattr(self._client, "generate"):
                # Some versions of the ollama client don't accept max_tokens
                # or temperature as keyword arguments on generate(); keep the
                # call minimal and pass only the model and prompt.
                resp = self._client.generate(
                    model=self.model_name,
                    prompt=prompt,
                )
            elif hasattr(self._client, "chat"):
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ]
                # Call chat with model and messages only to remain compatible
                # across client versions.
                resp = self._client.chat(
                    model=self.model_name,
                    messages=messages,
                )
            else:
                # Try a generic call
                resp = self._client.chat(self.model_name, prompt)
        except Exception as e:
            raise LLMError(f"Ollama generation failed: {e}")

        # Extract text from the response object in a defensive way
        def _extract(obj):
            # Attribute-style quick checks: common client responses may expose
            # `response`, `text`, `output`, or `result` attributes directly.
            try:
                for attr in ("response", "text", "output", "result"):
                    val = getattr(obj, attr, None)
                    if val:
                        return str(val)
            except Exception:
                pass

            # attribute-style message content (fallback)
            try:
                msg = getattr(obj, "message", None)
                if msg is not None:
                    content = getattr(msg, "content", None)
                    if content:
                        return str(content)
            except Exception:
                pass

            # dict-like
            try:
                if isinstance(obj, dict):
                    # common keys
                    for key in ("text", "output", "result", "response"):
                        if key in obj and obj[key]:
                            return str(obj[key])
                    if "message" in obj and isinstance(obj["message"], dict):
                        c = obj["message"].get("content") or obj["message"].get("text")
                        if c:
                            return str(c)
                    if "choices" in obj and isinstance(obj["choices"], list) and obj["choices"]:
                        c0 = obj["choices"][0]
                        if isinstance(c0, dict):
                            return str(c0.get("text") or c0.get("message") or "")
                        return str(c0)
            except Exception:
                pass

            # string-like
            if isinstance(obj, str):
                return obj

            # fallback to str()
            try:
                s = str(obj)
                if s:
                    return s
            except Exception:
                pass

            return None

        extracted = _extract(resp)
        if extracted:
            return extracted

        # If extraction failed, error so callers can fall back if desired
        raise LLMError("Ollama returned no textual output")

    def generate(self, prompt: str, max_tokens: int = None, temperature: float = None) -> str:
        """Compatibility wrapper: accept a single-string prompt and return
        the model output. This forwards to `generate_from_chunks` using a
        template that places the prompt as the `{context}`.
        """
        return self.generate_from_chunks(
            chunks=[prompt],
            prompt_template="{context}",
            question="",
            max_tokens=max_tokens,
            temperature=temperature,
        )


def get_llm(cfg: dict) -> Optional[LLM]:
    """Factory: returns an LLM instance or None if disabled/not configured."""
    if not cfg:
        return None

    llm_cfg = cfg.get("models", {}).get("llm") if isinstance(cfg, dict) else None
    if not llm_cfg:
        return None

    if not llm_cfg.get("enabled", False):
        return None

    provider = llm_cfg.get("provider", "gpt4all")

    if provider == "gpt4all":
        # Prefer an explicit `model` identifier (e.g. model name) if present,
        # otherwise fall back to `model_path` for backward compatibility.
        model_identifier = (
            llm_cfg.get("model")
            or llm_cfg.get("model_path")
            or "Mistral-7B-Instruct-v0.2-GGUF"
        )

        adapter = GPT4AllAdapter(
            model_path=model_identifier,
            n_ctx=llm_cfg.get("n_ctx", 2048),
            device=llm_cfg.get("device", "cpu"),
            max_tokens=llm_cfg.get("max_tokens", 512),
            temperature=llm_cfg.get("temperature", 0.0),
            auto_download=llm_cfg.get("auto_download", True),
            model_url=llm_cfg.get("model_url"),
            model_sha256=llm_cfg.get("model_sha256"),
        )

        # Force loading at factory time so callers know immediately
        # whether the model is available. This makes the model
        # effectively required (not optional) when enabled.
        adapter._load()
        return adapter
    elif provider == "ollama":
        model_identifier = llm_cfg.get("model") or llm_cfg.get("model_name")
        if not model_identifier:
            raise LLMError("Ollama provider requires 'model' configuration")

        adapter = OllamaAdapter(
            model_name=model_identifier,
            base_url=llm_cfg.get("base_url", "http://127.0.0.1:11434"),
            max_tokens=llm_cfg.get("max_tokens", 512),
            temperature=llm_cfg.get("temperature", 0.0),
        )

        # Verify that Ollama is reachable at factory time
        adapter._load()
        return adapter
    else:
        # Unknown provider: future extension point
        raise LLMError(f"Unsupported LLM provider: {provider}")
