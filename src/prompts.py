"""Prompt helpers for RAG templates."""
from pathlib import Path


def load_prompt(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    with open(p, 'r', encoding='utf-8') as f:
        return f.read()
