from retrieval.fusion import reciprocal_rank_fusion


def result(chunk_id: str, rank: int, article_id: str | None = None) -> dict:
    return {
        "rank": rank,
        "chunk_id": chunk_id,
        "article_id": article_id or f"article-{chunk_id}",
        "title": f"Title {chunk_id}",
        "source": "example.com",
        "url": f"https://example.com/{chunk_id}",
        "published_at": "2026-09-14T00:00:00+00:00",
        "text": f"Text for {chunk_id}",
    }


def test_rrf_combines_results_by_chunk_id_and_rank() -> None:
    bm25_results = [result("A", 1), result("B", 2), result("C", 3)]
    faiss_results = [result("C", 1), result("A", 2), result("D", 3)]

    fused = reciprocal_rank_fusion([bm25_results, faiss_results])
    by_id = {item["chunk_id"]: item for item in fused}

    assert by_id["A"]["rrf_score"] > by_id["B"]["rrf_score"]
    assert by_id["C"]["rrf_score"] > by_id["D"]["rrf_score"]
    assert by_id["A"]["retrieval_sources"] == {
        "bm25_rank": 1,
        "faiss_rank": 2,
    }
    assert by_id["C"]["retrieval_sources"] == {
        "bm25_rank": 3,
        "faiss_rank": 1,
    }
    assert [item["rank"] for item in fused] == list(range(1, 5))


def test_rrf_preserves_required_metadata() -> None:
    fused = reciprocal_rank_fusion([[result("A", 1)], []])

    assert fused[0]["chunk_id"] == "A"
    assert fused[0]["rrf_score"] == 1 / 61
    assert fused[0]["retrieval_sources"] == {
        "bm25_rank": 1,
        "faiss_rank": None,
    }
    assert all(
        field in fused[0]
        for field in (
            "rank",
            "article_id",
            "title",
            "source",
            "url",
            "published_at",
            "text",
        )
    )


def test_rrf_handles_empty_results() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_ignores_retriever_scores_and_duplicate_ids() -> None:
    first = result("A", 1)
    first["score"] = -1000
    duplicate = result("A", 2)
    duplicate["score"] = 1000
    second = result("B", 1)
    second["score"] = 0

    fused = reciprocal_rank_fusion([[first, duplicate], [second]])
    by_id = {item["chunk_id"]: item for item in fused}

    assert by_id["A"]["rrf_score"] == 1 / 61
    assert by_id["A"]["retrieval_sources"] == {
        "bm25_rank": 1,
        "faiss_rank": None,
    }
    assert "score" not in by_id["A"]
