"""Deterministic evidence sufficiency checks."""

from numbers import Real
from typing import Any


def assess_evidence(
    evidence: list[dict[str, Any]],
    min_items: int = 2,
) -> dict[str, Any]:
    """Assess count, text validity, rerank ordering, and article diversity."""

    if min_items <= 0:
        raise ValueError("min_items must be positive")

    valid = [
        item
        for item in evidence
        if isinstance(item, dict)
        and str(item.get("text", "")).strip()
        and str(item.get("article_id", "")).strip()
        and isinstance(item.get("rerank_score"), Real)
    ]
    ranked = sorted(valid, key=lambda item: float(item["rerank_score"]), reverse=True)
    unique_articles = {str(item["article_id"]) for item in ranked}
    required_articles = min(2, min_items)

    missing_facts: list[str] = []
    if len(ranked) < min_items:
        missing_facts.append(f"at least {min_items} valid evidence items")
    if len(unique_articles) < required_articles:
        missing_facts.append(f"evidence from at least {required_articles} distinct articles")

    if missing_facts:
        return {
            "sufficient": False,
            "reason": "; ".join(missing_facts) + " required",
            "missing_facts": missing_facts,
        }
    return {
        "sufficient": True,
        "reason": (
            f"Found {len(ranked)} valid reranked items from "
            f"{len(unique_articles)} distinct articles"
        ),
        "missing_facts": [],
    }
