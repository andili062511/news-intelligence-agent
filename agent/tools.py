"""Thin tool wrapper around the existing hybrid reranking pipeline."""

from functools import lru_cache
from typing import Any


EVIDENCE_FIELDS = (
    "chunk_id",
    "article_id",
    "title",
    "source",
    "url",
    "published_at",
    "text",
    "rrf_score",
    "rerank_score",
)


@lru_cache(maxsize=1)
def _default_retriever() -> Any:
    """Construct expensive embedding and reranking models only when used."""

    from retrieval.embeddings import EmbeddingModel
    from retrieval.faiss_retriever import FAISSRetriever
    from retrieval.hybrid import HybridRetriever
    from retrieval.loader import load_chunks
    from retrieval.reranker import CrossEncoderReranker, HybridRerankRetriever

    chunks = load_chunks()
    hybrid = HybridRetriever(
        chunks,
        faiss_retriever=FAISSRetriever(chunks, encoder=EmbeddingModel()),
    )
    return HybridRerankRetriever(hybrid, CrossEncoderReranker())


def search_news_tool(
    query: str,
    top_k: int = 5,
    candidate_k: int = 15,
) -> list[dict[str, Any]]:
    """Return evidence from the existing Hybrid + Cross-Encoder pipeline."""

    results = _default_retriever().search(
        query,
        top_k=top_k,
        candidate_k=candidate_k,
    )
    return [
        {field: result.get(field) for field in EVIDENCE_FIELDS}
        for result in results
    ]
