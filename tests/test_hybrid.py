from typing import Any

from retrieval.hybrid import HybridRetriever


CHUNKS = [
    {
        "chunk_id": "chunk-1",
        "article_id": "article-1",
        "title": "AI safety article",
        "source": "example.com",
        "url": "https://example.com/1",
        "published_at": "2026-09-14T00:00:00+00:00",
        "text": "AI safety and risks.",
    },
    {
        "chunk_id": "chunk-2",
        "article_id": "article-1",
        "title": "AI safety continuation",
        "source": "example.com",
        "url": "https://example.com/2",
        "published_at": "2026-09-14T00:00:00+00:00",
        "text": "More details about AI safety.",
    },
    {
        "chunk_id": "chunk-3",
        "article_id": "article-2",
        "title": "AI policy article",
        "source": "another.example.com",
        "url": "https://another.example.com/3",
        "published_at": "2026-09-14T00:00:00+00:00",
        "text": "AI policy and governance.",
    },
]


def retrieval_result(chunk: dict[str, Any], rank: int) -> dict[str, Any]:
    return {"rank": rank, **chunk}


class FakeRetriever:
    def __init__(self, results: list[dict[str, Any]]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        self.calls.append((query, top_k))
        return self.results[:top_k]


def test_hybrid_calls_both_retrievers_and_returns_fused_results() -> None:
    bm25 = FakeRetriever([retrieval_result(CHUNKS[0], 1), retrieval_result(CHUNKS[2], 2)])
    faiss = FakeRetriever([retrieval_result(CHUNKS[2], 1), retrieval_result(CHUNKS[1], 2)])
    retriever = HybridRetriever(CHUNKS, bm25_retriever=bm25, faiss_retriever=faiss)

    results = retriever.search("AI safety", top_k=2, candidate_k=1)

    assert bm25.calls == [("AI safety", 2)]
    assert faiss.calls == [("AI safety", 2)]
    assert len(results) == 2
    assert results[0]["chunk_id"] == "chunk-3"
    assert results[0]["retrieval_sources"] == {"bm25_rank": 2, "faiss_rank": 1}
    assert [item["rank"] for item in results] == [1, 2]


def test_hybrid_handles_empty_query() -> None:
    bm25 = FakeRetriever([])
    faiss = FakeRetriever([])
    retriever = HybridRetriever(CHUNKS, bm25_retriever=bm25, faiss_retriever=faiss)

    assert retriever.search("   ") == []
    assert bm25.calls == []
    assert faiss.calls == []


def test_hybrid_handles_empty_corpus() -> None:
    assert HybridRetriever([]).search("AI safety") == []


def test_hybrid_limits_chunks_per_article_after_fusion() -> None:
    bm25 = FakeRetriever(
        [
            retrieval_result(CHUNKS[0], 1),
            retrieval_result(CHUNKS[1], 2),
            retrieval_result(CHUNKS[2], 3),
        ]
    )
    faiss = FakeRetriever(
        [
            retrieval_result(CHUNKS[0], 1),
            retrieval_result(CHUNKS[1], 2),
            retrieval_result(CHUNKS[2], 3),
        ]
    )
    retriever = HybridRetriever(CHUNKS, bm25_retriever=bm25, faiss_retriever=faiss)

    results = retriever.search(
        "AI safety",
        top_k=2,
        candidate_k=3,
        max_chunks_per_article=1,
    )

    assert [item["chunk_id"] for item in results] == ["chunk-1", "chunk-3"]
    assert len({item["article_id"] for item in results}) == len(results)


def test_hybrid_reports_missing_retrieval_route() -> None:
    bm25 = FakeRetriever([retrieval_result(CHUNKS[0], 1)])
    faiss = FakeRetriever([])
    retriever = HybridRetriever(CHUNKS, bm25_retriever=bm25, faiss_retriever=faiss)

    results = retriever.search("AI safety", top_k=1)

    assert results[0]["retrieval_sources"] == {
        "bm25_rank": 1,
        "faiss_rank": None,
    }
