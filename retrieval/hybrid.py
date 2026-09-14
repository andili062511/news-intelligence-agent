"""Hybrid BM25 and FAISS retrieval with Reciprocal Rank Fusion."""

from typing import Any

from .bm25 import BM25Retriever
from .faiss_retriever import FAISSRetriever
from .fusion import reciprocal_rank_fusion


class HybridRetriever:
    """Combine lexical and dense retrieval results with RRF."""

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        bm25_retriever: BM25Retriever | Any | None = None,
        faiss_retriever: FAISSRetriever | Any | None = None,
        rrf_k: int = 60,
    ) -> None:
        self.chunks = chunks
        self.rrf_k = rrf_k
        self.bm25_retriever = (
            bm25_retriever if bm25_retriever is not None else BM25Retriever(chunks)
        )

        if faiss_retriever is not None:
            self.faiss_retriever = faiss_retriever
        else:
            # FAISSRetriever skips model loading for an empty corpus.
            self.faiss_retriever = FAISSRetriever(chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 10,
        max_chunks_per_article: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return fused results, optionally limiting chunks per article."""

        if not query or not query.strip() or top_k <= 0 or not self.chunks:
            return []
        if max_chunks_per_article is not None and max_chunks_per_article <= 0:
            return []

        candidate_k = max(candidate_k, top_k)
        bm25_results = self.bm25_retriever.search(query, top_k=candidate_k)
        faiss_results = (
            self.faiss_retriever.search(query, top_k=candidate_k)
            if self.faiss_retriever is not None
            else []
        )
        fused_results = reciprocal_rank_fusion(
            [bm25_results, faiss_results],
            k=self.rrf_k,
        )

        selected: list[dict[str, Any]] = []
        article_counts: dict[Any, int] = {}
        for result in fused_results:
            if max_chunks_per_article is not None:
                article_id = result.get("article_id", "")
                if article_counts.get(article_id, 0) >= max_chunks_per_article:
                    continue
                article_counts[article_id] = article_counts.get(article_id, 0) + 1

            selected.append(dict(result))
            if len(selected) >= top_k:
                break

        for rank, result in enumerate(selected, start=1):
            result["rank"] = rank
        return selected
