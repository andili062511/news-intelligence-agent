from typing import Any

from retrieval.reranker import CrossEncoderReranker, HybridRerankRetriever


def candidate(
    chunk_id: str,
    article_id: str,
    text: str,
    rrf_score: float,
    rank: int,
) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "article_id": article_id,
        "title": f"Title {chunk_id}",
        "source": "example.com",
        "url": f"https://example.com/{chunk_id}",
        "published_at": "2026-09-14T00:00:00+00:00",
        "text": text,
        "rrf_score": rrf_score,
        "retrieval_sources": {"bm25_rank": rank, "faiss_rank": rank},
        "rank": rank,
    }


class FakeCrossEncoder:
    def __init__(self, scores_by_text: dict[str, float]) -> None:
        self.scores_by_text = scores_by_text
        self.calls: list[list[list[str]]] = []

    def predict(self, pairs: list[list[str]]) -> list[float]:
        self.calls.append(pairs)
        return [
            next(score for text, score in self.scores_by_text.items() if text in document)
            for _, document in pairs
        ]


class FakeHybridRetriever:
    def __init__(self, results: list[dict[str, Any]]) -> None:
        self.results = results
        self.calls: list[tuple[str, int, int]] = []

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 10,
        max_chunks_per_article: int | None = None,
    ) -> list[dict[str, Any]]:
        self.calls.append((query, top_k, candidate_k))
        return self.results[:top_k]


CANDIDATES = [
    candidate("chunk-1", "article-1", "low relevance", 0.04, 1),
    candidate("chunk-2", "article-1", "highest relevance", 0.03, 2),
    candidate("chunk-3", "article-2", "medium relevance", 0.02, 3),
]


def make_reranker() -> tuple[CrossEncoderReranker, FakeCrossEncoder]:
    model = FakeCrossEncoder(
        {"low relevance": 0.1, "highest relevance": 0.9, "medium relevance": 0.5}
    )
    return CrossEncoderReranker(model=model), model


def test_rerank_changes_hybrid_order_and_adds_scores() -> None:
    reranker, _ = make_reranker()

    results = reranker.rerank("relevant query", CANDIDATES, top_k=3)

    assert [result["chunk_id"] for result in results] == [
        "chunk-2",
        "chunk-3",
        "chunk-1",
    ]
    assert [result["rerank_score"] for result in results] == [0.9, 0.5, 0.1]
    assert [result["rank"] for result in results] == [1, 2, 3]
    assert results[0]["rrf_score"] == 0.03
    assert results[0]["retrieval_sources"] == {"bm25_rank": 2, "faiss_rank": 2}


def test_rerank_does_not_modify_original_candidates() -> None:
    reranker, _ = make_reranker()
    original = [dict(item) for item in CANDIDATES]

    reranker.rerank("relevant query", CANDIDATES, top_k=3)

    assert CANDIDATES == original
    assert all("rerank_score" not in item for item in CANDIDATES)


def test_rerank_handles_empty_inputs_and_non_positive_top_k() -> None:
    reranker, model = make_reranker()

    assert reranker.rerank("   ", CANDIDATES) == []
    assert reranker.rerank("query", []) == []
    assert reranker.rerank("query", CANDIDATES, top_k=0) == []
    assert model.calls == []


def test_rerank_top_k_and_candidate_count_limits() -> None:
    reranker, _ = make_reranker()

    assert len(reranker.rerank("query", CANDIDATES, top_k=2)) == 2
    assert len(reranker.rerank("query", CANDIDATES, top_k=20)) == 3


def test_rerank_predicts_all_pairs_in_one_batch() -> None:
    reranker, model = make_reranker()

    reranker.rerank("query", CANDIDATES, top_k=3)

    assert len(model.calls) == 1
    assert len(model.calls[0]) == 3
    assert model.calls[0][0] == ["query", "Title chunk-1 low relevance"]


def test_pipeline_applies_article_limit_after_reranking() -> None:
    reranker, _ = make_reranker()
    hybrid = FakeHybridRetriever(CANDIDATES)
    pipeline = HybridRerankRetriever(hybrid, reranker)

    results = pipeline.search(
        "query",
        top_k=2,
        candidate_k=3,
        max_chunks_per_article=1,
    )

    assert hybrid.calls == [("query", 3, 3)]
    assert [result["chunk_id"] for result in results] == ["chunk-2", "chunk-3"]
    assert [result["rank"] for result in results] == [1, 2]
    assert len({result["article_id"] for result in results}) == 2
