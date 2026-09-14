"""Deterministic citation validation for generated answers."""

import re
from collections.abc import Mapping
from typing import Any


_CITATION_PATTERN = re.compile(r"\[(E\d+)\]")


def extract_citation_ids(answer: str) -> list[str]:
    """Extract unique evidence IDs in first-appearance order."""

    return list(dict.fromkeys(_CITATION_PATTERN.findall(answer or "")))


def validate_answer_citations(
    answer: str,
    grounding_pack: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject empty, uncited, or unknown-citation generated answers."""

    if not isinstance(answer, str) or not answer.strip():
        return {
            "valid": False,
            "used_citations": [],
            "invalid_citations": [],
            "reason": "empty_answer",
        }

    used = extract_citation_ids(answer)
    allowed = {
        str(item.get("evidence_id"))
        for item in grounding_pack.get("evidence", [])
        if isinstance(item, Mapping) and item.get("evidence_id")
    }
    invalid = [citation for citation in used if citation not in allowed]
    if invalid:
        labels = ", ".join(f"[{citation}]" for citation in invalid)
        return {
            "valid": False,
            "used_citations": used,
            "invalid_citations": invalid,
            "reason": f"Unknown citation {labels}",
        }
    if not used:
        return {
            "valid": False,
            "used_citations": [],
            "invalid_citations": [],
            "reason": "factual_answer_without_citation",
        }
    return {
        "valid": True,
        "used_citations": used,
        "invalid_citations": [],
        "reason": "valid",
    }
