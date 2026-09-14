"""Parse, validate, and render structured grounded claims."""

import json
from collections.abc import Mapping
from typing import Any


def parse_claim_response(text: str) -> dict[str, Any]:
    """Parse the model response without raising on malformed output."""

    if not isinstance(text, str) or not text.strip():
        return {"valid": False, "errors": ["invalid_json: empty response"], "claims": []}
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        return {
            "valid": False,
            "errors": [f"invalid_json: {exc}"],
            "claims": [],
        }
    if not isinstance(payload, dict):
        return {
            "valid": False,
            "errors": ["invalid_schema: top-level value must be an object"],
            "claims": [],
        }
    claims = payload.get("claims")
    if not isinstance(claims, list):
        return {
            "valid": False,
            "errors": ["invalid_schema: top-level claims must be a list"],
            "claims": [],
        }

    errors: list[str] = []
    for index, claim in enumerate(claims, start=1):
        if not isinstance(claim, dict):
            errors.append(f"invalid_schema: claim {index} must be an object")
            continue
        if "text" not in claim:
            errors.append(f"invalid_schema: claim {index} is missing text")
        elif not isinstance(claim["text"], str) or not claim["text"].strip():
            errors.append(f"empty_claim: claim {index} text must not be empty")
        if "citations" not in claim:
            errors.append(f"missing_citation: claim {index} is missing citations")
        elif not isinstance(claim["citations"], list):
            errors.append(f"invalid_schema: claim {index} citations must be a list")
        elif not claim["citations"]:
            errors.append(f"missing_citation: claim {index} must cite evidence")
    return {"valid": not errors, "errors": errors, "claims": claims}


def validate_claims(
    claims: Any,
    grounding_pack: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate claim citations and return normalized, de-duplicated claims."""

    if not isinstance(claims, list):
        return {
            "valid": False,
            "errors": ["invalid_schema: claims must be a list"],
            "claims": [],
        }
    if not claims:
        return {
            "valid": False,
            "errors": ["missing_claims: claims must not be empty"],
            "claims": [],
        }

    allowed = {
        str(item.get("evidence_id"))
        for item in grounding_pack.get("evidence", [])
        if isinstance(item, Mapping) and item.get("evidence_id")
    }
    errors: list[str] = []
    normalized: list[dict[str, Any]] = []
    for index, claim in enumerate(claims, start=1):
        if not isinstance(claim, Mapping):
            errors.append(f"invalid_schema: claim {index} must be an object")
            continue
        text = claim.get("text")
        citations = claim.get("citations")
        clean_text = text.strip() if isinstance(text, str) else ""
        if not clean_text:
            errors.append(f"empty_claim: claim {index} text must not be empty")
        if not isinstance(citations, list):
            errors.append(f"invalid_schema: claim {index} citations must be a list")
            clean_citations: list[str] = []
        else:
            clean_citations = []
            for citation in citations:
                if not isinstance(citation, str) or not citation.strip():
                    errors.append(
                        f"invalid_citation: claim {index} contains a non-string or empty citation"
                    )
                    continue
                citation_id = citation.strip()
                if citation_id not in clean_citations:
                    clean_citations.append(citation_id)
        if not clean_citations:
            errors.append(f"missing_citation: claim {index} must cite evidence")
        for citation_id in clean_citations:
            if citation_id not in allowed:
                errors.append(
                    f"invalid_citation: claim {index} cites unknown evidence ID {citation_id}"
                )
        normalized.append({"text": clean_text, "citations": clean_citations})
    return {"valid": not errors, "errors": errors, "claims": normalized}


def render_claims(claims: list[dict[str, Any]]) -> str:
    """Render only the citations explicitly attached to each validated claim."""

    paragraphs: list[str] = []
    for claim in claims:
        text = str(claim["text"]).strip()
        citations = "".join(f"[{citation}]" for citation in claim["citations"])
        punctuation = text[-1] if text.endswith((".", "?", "!")) else ""
        body = text[:-1].rstrip() if punctuation else text
        paragraphs.append(f"{body} {citations}{punctuation}")
    return "\n\n".join(paragraphs)
