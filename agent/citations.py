"""Citation identifiers, metadata validation, and deterministic deduplication."""

from collections.abc import Mapping, Sequence
from typing import Any


REQUIRED_CITATION_FIELDS = (
    "evidence_id",
    "article_id",
    "chunk_id",
    "title",
    "source",
    "url",
    "text",
)


def assign_evidence_ids(
    evidence: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return copied evidence items labelled E1, E2, ... in ranking order."""

    return [
        {
            **(dict(item) if isinstance(item, Mapping) else {}),
            "evidence_id": f"E{index}",
        }
        for index, item in enumerate(evidence, start=1)
    ]


def validate_citation_metadata(
    evidence: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate required, non-empty citation fields without raising on bad data."""

    errors: list[dict[str, Any]] = []
    for index, item in enumerate(evidence, start=1):
        if not isinstance(item, Mapping):
            errors.append(
                {
                    "evidence_id": f"E{index}",
                    "reason": "evidence item is not an object",
                }
            )
            continue

        evidence_id = str(item.get("evidence_id") or f"E{index}").strip()
        for field in REQUIRED_CITATION_FIELDS:
            value = item.get(field)
            if value is None or not str(value).strip():
                errors.append(
                    {
                        "evidence_id": evidence_id,
                        "reason": f"missing {field}",
                    }
                )

    return {"valid": not errors, "errors": errors}


def detect_duplicate_evidence(
    evidence: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Remove exact citation duplicates while preserving the current ranking."""

    unique: list[dict[str, Any]] = []
    seen_chunks: set[str] = set()
    seen_article_text: set[tuple[str, str]] = set()
    seen_url_indexes: set[tuple[str, str]] = set()
    removed = 0

    for raw_item in evidence:
        item = dict(raw_item) if isinstance(raw_item, Mapping) else {}
        chunk_id = str(item.get("chunk_id") or "").strip()
        article_id = str(item.get("article_id") or "").strip()
        text = str(item.get("text") or "").strip()
        url = str(item.get("url") or "").strip()
        chunk_index_value = item.get("chunk_index")
        has_chunk_index = (
            chunk_index_value is not None
            and str(chunk_index_value).strip() != ""
        )
        url_index = (url, str(chunk_index_value))

        duplicate = (
            (bool(chunk_id) and chunk_id in seen_chunks)
            or (
                bool(article_id)
                and bool(text)
                and (article_id, text) in seen_article_text
            )
            or (
                bool(url)
                and has_chunk_index
                and url_index in seen_url_indexes
            )
        )
        if duplicate:
            removed += 1
            continue

        unique.append(item)
        if chunk_id:
            seen_chunks.add(chunk_id)
        if article_id and text:
            seen_article_text.add((article_id, text))
        if url and has_chunk_index:
            seen_url_indexes.add(url_index)

    return unique, removed
