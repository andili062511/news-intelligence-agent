"""BM25 retrieval over news chunks."""

from typing import Any

from rank_bm25 import BM25Okapi

from .tokenizer import tokenize


class BM25Retriever:
    """Rank news chunks using BM25 lexical matching."""

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks
        corpus = []
        for chunk in chunks:
            title = str(chunk.get("title") or "")
            text = str(chunk.get("text") or "")
            document_text = title + " " + text
            corpus.append(tokenize(document_text))
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return up to ``top_k`` chunks ordered by descending BM25 score."""

        if not query or not query.strip() or top_k <= 0 or not self.chunks:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        ranked_indices = sorted(
            range(len(self.chunks)),
            key=lambda index: float(scores[index]),
            reverse=True,
        )[:top_k]

        results: list[dict[str, Any]] = []
        for rank, index in enumerate(ranked_indices, start=1):
            chunk = self.chunks[index]
            results.append(
                {
                    "rank": rank,
                    "score": float(scores[index]),
                    "chunk_id": chunk.get("chunk_id", ""),
                    "article_id": chunk.get("article_id", ""),
                    "title": chunk.get("title", ""),
                    "source": chunk.get("source", ""),
                    "url": chunk.get("url", ""),
                    "published_at": chunk.get("published_at", ""),
                    "text": chunk.get("text", ""),
                }
            )
        return results
