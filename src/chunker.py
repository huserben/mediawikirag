"""Text chunking logic for Wiki RAG.

Provides a recursive/sentence-aware `chunk_text` implementation that aims
to preserve paragraph and sentence boundaries where possible, falling back
to fixed-size slicing when necessary. The function keeps the same public
signature used by tests and other code.
"""

import re
from typing import List, Dict


def _split_sentences(text: str) -> List[str]:
    # Simple sentence splitter using punctuation boundaries. Falls back to
    # returning the whole text if no sentence boundaries are found.
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) <= 1:
        return [text]
    return sentences


def chunk_text(page: Dict, chunk_size: int = 800, overlap: int = 150) -> List[Dict]:
    """Split `page['content']` into chunks trying to keep sentences intact.

    Algorithm:
    - Split text into paragraphs on two-or-more newlines.
    - Accumulate paragraphs into a current chunk until adding the next
      paragraph would exceed `chunk_size`.
    - If a single paragraph is longer than `chunk_size`, split it into
      sentences and assemble chunks from sentences.
    - If sentences are still longer than `chunk_size` (e.g. long unbroken
      text), fall back to fixed-size slicing with `overlap`.

    Returns a list of chunk dicts with keys: `id`, `text`, `page_title`,
    `chunk_index`, `char_count`.
    """
    text = page.get('content', '') or ''
    title = page.get('title', '') or ''

    if not text:
        return []

    paragraphs = re.split(r'\n{2,}', text)
    chunks = []
    current = ''
    idx = 0

    def _emit(part: str):
        nonlocal idx
        part = part.strip()
        if not part:
            return
        chunk = {
            'id': f"{title}_chunk{idx:03d}",
            'text': part,
            'page_title': title,
            'chunk_index': idx,
            'char_count': len(part)
        }
        chunks.append(chunk)
        idx += 1

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If paragraph fits into chunk_size, try to append to current
        if len(para) <= chunk_size:
            if not current:
                current = para
            else:
                # If joining would exceed size, emit current and start new
                if len(current) + 2 + len(para) > chunk_size:
                    # emit current (with overlap preserved for next)
                    _emit(current)
                    # keep overlap tail
                    tail = current[-overlap:] if overlap and len(current) > overlap else ''
                    current = (tail + '\n\n' + para).strip()
                else:
                    current = (current + '\n\n' + para).strip()
        else:
            # Paragraph too large; split into sentences and attempt to assemble
            sentences = _split_sentences(para)
            # If splitting produced only one long sentence, fallback to slicing
            if len(sentences) == 1 and len(sentences[0]) > chunk_size:
                # flush current
                if current:
                    _emit(current)
                    current = ''
                long_text = sentences[0]
                start = 0
                while start < len(long_text):
                    end = min(start + chunk_size, len(long_text))
                    slice_part = long_text[start:end]
                    _emit(slice_part)
                    start += chunk_size - overlap
                continue

            # assemble chunks from sentences
            for s in sentences:
                s = s.strip()
                if not s:
                    continue
                if not current:
                    current = s
                else:
                    if len(current) + 1 + len(s) > chunk_size:
                        _emit(current)
                        tail = current[-overlap:] if overlap and len(current) > overlap else ''
                        current = (tail + ' ' + s).strip()
                    else:
                        current = (current + ' ' + s).strip()

    # emit final buffer
    if current:
        _emit(current)

    return chunks
