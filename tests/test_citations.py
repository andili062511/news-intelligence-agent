from copy import deepcopy
from typing import Any

from agent.citations import (
    assign_evidence_ids,
    detect_duplicate_evidence,
    validate_citation_metadata,
)


def citation(number: int = 1, **overrides: Any) -> dict[str, Any]:
    item = {
        "chunk_id": f"chunk-{number}",
        "article_id": f"article-{number}",
        "title": f"Title {number}",
        "source": "Example News",
        "url": f"https://example.com/{number}",
        "published_at": "2026-09-14T00:00:00+00:00",
        "text": f"Evidence {number}",
        "rerank_score": 1.0 / number,
    }
    item.update(overrides)
    return item


def test_assign_evidence_ids_preserves_ranking_order() -> None:
    result = assign_evidence_ids([citation(3), citation(1), citation(2)])

    assert [item["evidence_id"] for item in result] == ["E1", "E2", "E3"]
    assert [item["article_id"] for item in result] == [
        "article-3",
        "article-1",
        "article-2",
    ]
    assert len({item["evidence_id"] for item in result}) == 3


def test_assign_evidence_ids_does_not_modify_input() -> None:
    evidence = [citation(1), citation(2, evidence_id="old")]
    original = deepcopy(evidence)

    result = assign_evidence_ids(evidence)

    assert evidence == original
    assert result is not evidence
    assert result[0] is not evidence[0]


def test_missing_url_and_text_are_reported_without_raising() -> None:
    identified = assign_evidence_ids(
        [citation(1, url=""), citation(2, text=None)]
    )

    result = validate_citation_metadata(identified)

    assert result["valid"] is False
    assert {error["reason"] for error in result["errors"]} == {
        "missing url",
        "missing text",
    }
    assert {error["evidence_id"] for error in result["errors"]} == {"E1", "E2"}


def test_duplicate_chunk_id_is_removed_without_mutating_input() -> None:
    evidence = [citation(1), citation(2, chunk_id="chunk-1")]
    original = deepcopy(evidence)

    deduplicated, removed = detect_duplicate_evidence(evidence)

    assert removed == 1
    assert deduplicated == [evidence[0]]
    assert evidence == original
    assert deduplicated[0] is not evidence[0]


def test_duplicate_article_text_and_url_chunk_index_are_removed() -> None:
    evidence = [
        citation(1, chunk_index=0),
        citation(2, text="Evidence 1", article_id="article-1"),
        citation(3, url="https://example.com/1", chunk_index=0),
    ]

    deduplicated, removed = detect_duplicate_evidence(evidence)

    assert len(deduplicated) == 1
    assert removed == 2
