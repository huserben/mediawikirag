# Text chunking logic for Wiki RAG

def chunk_text(page, chunk_size=800, overlap=150):
    """
    Splits page content into fixed-size chunks with overlap.
    Adds metadata for each chunk.
    Args:
        page (dict): {'title': str, 'content': str}
    Returns:
        List[dict]: [{'id', 'text', 'page_title', 'chunk_index', 'char_count'}]
    """
    text = page.get('content', '')
    title = page.get('title', '')
    chunks = []
    start = 0
    chunk_index = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]
        chunk = {
            'id': f"{title}_chunk{chunk_index:03d}",
            'text': chunk_text,
            'page_title': title,
            'chunk_index': chunk_index,
            'char_count': len(chunk_text)
        }
        chunks.append(chunk)
        start += chunk_size - overlap
        chunk_index += 1
    return chunks
