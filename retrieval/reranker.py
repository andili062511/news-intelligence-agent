"""Cross-encoder reranking for hybrid retrieval candidates."""

from typing import Any

from .hybrid import HybridRetriever


DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Score query-document pairs with a cross-encoder and reorder them."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        model: Any | None = None,
    ) -> None:
        self.model_name = model_name
        if model is None:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(model_name)
        self.model = model

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return the highest-scoring copies of the supplied candidates."""

        if not query or not query.strip() or not candidates or top_k <= 0:
            return []

        pairs = [
            [query, f"{candidate.get('title', '')} {candidate.get('text', '')}".strip()]
            for candidate in candidates
        ]
        scores = self.model.predict(pairs)
        if len(scores) != len(candidates):
            raise ValueError("Cross-encoder returned a different number of scores")

        scored = []
        for candidate, score in zip(candidates, scores):
            result = dict(candidate)
            result["rerank_score"] = float(score)
            scored.append(result)

        scored.sort(key=lambda result: result["rerank_score"], reverse=True)
        selected = scored[:top_k]
        for rank, result in enumerate(selected, start=1):
            result["rank"] = rank
        return selected


class HybridRerankRetriever:
    """Retrieve broad hybrid candidates, then rerank and diversify them."""

    def __init__(
        self,
        hybrid_retriever: HybridRetriever | Any,
        reranker: CrossEncoderReranker,
    ) -> None:
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 15,
        max_chunks_per_article: int | None = None,
    ) -> list[dict[str, Any]]:
        """Run hybrid retrieval, reranking, diversity filtering, and Top-K."""

        if not query or not query.strip() or top_k <= 0 or candidate_k <= 0:
            return []
        if max_chunks_per_article is not None and max_chunks_per_article <= 0:
            return []

        candidates = self.hybrid_retriever.search(
            query,
            top_k=candidate_k,
            candidate_k=candidate_k,
        )
        reranked = self.reranker.rerank(
            query,
            candidates,
            top_k=len(candidates),
        )

        selected: list[dict[str, Any]] = []
        article_counts: dict[Any, int] = {}
        for result in reranked:
            if max_chunks_per_article is not None:
                article_id = result.get("article_id", "")
                if article_counts.get(article_id, 0) >= max_chunks_per_article:
                    continue
                article_counts[article_id] = article_counts.get(article_id, 0) + 1

            selected.append(result)
            if len(selected) >= top_k:
                break

        for rank, result in enumerate(selected, start=1):
            result["rank"] = rank
        return selected
