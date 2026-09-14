from typing import Any

import pytest

from agent.evidence import assess_evidence


def evidence_item(article_id: str, score: float, text: str = "Useful report") -> dict[str, Any]:
    return {
        "chunk_id": f"chunk-{article_id}-{score}",
        "article_id": article_id,
        "text": text,
        "rerank_score": score,
    }


def test_evidence_is_sufficient_with_valid_diverse_articles() -> None:
    result = assess_evidence(
        [evidence_item("article-1", -3.0), evidence_item("article-2", -9.0)]
    )

    assert result["sufficient"] is True
    assert result["missing_facts"] == []


def test_evidence_is_insufficient_when_too_few_items_or_text_is_empty() -> None:
    result = assess_evidence(
        [evidence_item("article-1", 100.0), evidence_item("article-2", 90.0, " ")]
    )

    assert result["sufficient"] is False
    assert result["missing_facts"]


def test_chunks_from_only_one_article_are_insufficient() -> None:
    result = assess_evidence(
        [evidence_item("article-1", 0.8), evidence_item("article-1", 0.7)]
    )

    assert result["sufficient"] is False
    assert "distinct articles" in result["reason"]


def test_evidence_does_not_depend_on_absolute_score_scale_or_input_order() -> None:
    result = assess_evidence(
        [evidence_item("article-1", -25.0), evidence_item("article-2", -1.0)]
    )

    assert result["sufficient"] is True


def test_min_items_must_be_positive() -> None:
    with pytest.raises(ValueError):
        assess_evidence([], min_items=0)
