"""
Lightweight LLM adapter layer.
Provides a simple interface and a GPT4All adapter with optional
auto-download support via `src.download.ensure_model`.
"""

from typing import Optional
from gpt4all import GPT4All  # Fixed: lowercase 'gpt4all'
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
    else:
        # Unknown provider: future extension point
        raise LLMError(f"Unsupported LLM provider: {provider}")
