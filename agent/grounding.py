"""Build and assess evidence packs that are safe to pass to a future LLM."""

from collections.abc import Mapping, Sequence
from typing import Any

from .citations import (
    assign_evidence_ids,
    detect_duplicate_evidence,
    validate_citation_metadata,
)


GROUNDING_FIELDS = (
    "evidence_id",
    "chunk_id",
    "article_id",
    "title",
    "source",
    "url",
    "published_at",
    "text",
)


def build_grounding_pack(
    question: str,
    evidence: Sequence[Mapping[str, Any]],
    max_items: int = 3,
    max_chunks_per_article: int | None = 1,
) -> dict[str, Any]:
    """Select, diversify, label, validate, and project ranked evidence.

    ``max_items`` defines the ranked candidate window as well as the output
    ceiling.  Article diversity is applied inside that window, so weak items
    below it are not pulled in merely to fill the pack.
    """

    if max_items < 0:
        raise ValueError("max_items must be non-negative")
    if max_chunks_per_article is not None and max_chunks_per_article <= 0:
        raise ValueError("max_chunks_per_article must be positive or None")

    if max_items == 0:
        return {
            "question": str(question),
            "evidence_count": 0,
            "evidence": [],
            "citation_errors": [],
        }

    ranked_window = evidence[:max_items]
    deduplicated, _ = detect_duplicate_evidence(ranked_window)
    diversified: list[dict[str, Any]] = []
    article_counts: dict[str, int] = {}
    for item in deduplicated:
        article_id = str(item.get("article_id") or "").strip()
        if max_chunks_per_article is not None:
            count = article_counts.get(article_id, 0)
            if count >= max_chunks_per_article:
                continue
            article_counts[article_id] = count + 1
        diversified.append(item)

    valid_items: list[dict[str, Any]] = []
    citation_errors: list[dict[str, Any]] = []
    for item in assign_evidence_ids(diversified):
        validation = validate_citation_metadata([item])
        if not validation["valid"]:
            citation_errors.extend(validation["errors"])
            continue
        if not str(item.get("text") or "").strip():
            continue
        valid_items.append({field: item.get(field) for field in GROUNDING_FIELDS})

    return {
        "question": str(question),
        "evidence_count": len(valid_items),
        "evidence": valid_items,
        "citation_errors": citation_errors,
    }


def assess_grounding_readiness(
    grounding_pack: Mapping[str, Any],
    min_evidence: int = 2,
) -> dict[str, Any]:
    """Require enough valid citations from multiple source articles."""

    if min_evidence <= 0:
        raise ValueError("min_evidence must be positive")

    raw_evidence = grounding_pack.get("evidence", [])
    evidence = raw_evidence if isinstance(raw_evidence, list) else []
    valid = [
        item
        for item in evidence
        if isinstance(item, Mapping)
        and validate_citation_metadata([item])["valid"]
        and str(item.get("text") or "").strip()
    ]
    unique_articles = {
        str(item.get("article_id") or "").strip()
        for item in valid
        if str(item.get("article_id") or "").strip()
    }

    if len(valid) < min_evidence:
        return {
            "ready": False,
            "reason": f"at least {min_evidence} valid evidence items required",
            "evidence_count": len(valid),
            "unique_articles": len(unique_articles),
        }
    if len(unique_articles) < 2:
        return {
            "ready": False,
            "reason": (
                "evidence from at least 2 distinct articles required for "
                "a multi-source answer"
            ),
            "evidence_count": len(valid),
            "unique_articles": len(unique_articles),
        }
    return {
        "ready": True,
        "reason": (
            f"Found {len(valid)} valid citations from "
            f"{len(unique_articles)} distinct articles"
        ),
        "evidence_count": len(valid),
        "unique_articles": len(unique_articles),
    }
