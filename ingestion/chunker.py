"""Word-based article chunking."""

from typing import Any


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    """Split text into overlapping word chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    words = text.split() if text else []
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def create_chunks(article: dict[str, Any]) -> list[dict[str, Any]]:
    """Create chunk records while carrying article metadata into each chunk."""

    article_id = str(article.get("article_id") or "")
    chunks = chunk_text(str(article.get("text") or ""))
    return [
        {
            "chunk_id": f"{article_id}_{index}",
            "article_id": article_id,
            "chunk_index": index,
            "title": article.get("title", ""),
            "source": article.get("source", ""),
            "url": article.get("url", ""),
            "published_at": article.get("published_at", ""),
            "text": text,
        }
        for index, text in enumerate(chunks)
    ]

