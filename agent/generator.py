"""Grounding gate and model-independent answer generation."""

from collections.abc import Mapping
from typing import Any

from llm.base import BaseLLM
from llm.prompts import build_citation_repair_messages, build_grounded_messages

from .citations import validate_citation_metadata


def generate_grounded_answer(
    question: str,
    grounding_pack: Mapping[str, Any],
    llm: BaseLLM,
) -> dict[str, Any]:
    """Generate only when the graph has marked its validated pack ready."""

    evidence = grounding_pack.get("evidence", [])
    citations_valid = (
        isinstance(evidence, list)
        and bool(evidence)
        and validate_citation_metadata(evidence)["valid"]
    )
    if grounding_pack.get("ready") is not True or not citations_valid:
        return {
            "generated": False,
            "answer": None,
            "reason": "insufficient_grounding",
        }

    messages = build_grounded_messages(question, grounding_pack)
    answer = llm.generate(messages)
    evidence_ids = [
        str(item.get("evidence_id"))
        for item in grounding_pack.get("evidence", [])
        if isinstance(item, Mapping) and item.get("evidence_id")
    ]
    return {
        "generated": True,
        "answer": answer,
        "used_evidence_ids": evidence_ids,
    }


def repair_grounded_answer(
    question: str,
    grounding_pack: Mapping[str, Any],
    generated_answer: str,
    validation_error: str,
    llm: BaseLLM,
) -> dict[str, Any]:
    """Perform one citation-aware rewrite against the same validated pack."""

    evidence = grounding_pack.get("evidence", [])
    citations_valid = (
        isinstance(evidence, list)
        and bool(evidence)
        and validate_citation_metadata(evidence)["valid"]
    )
    if grounding_pack.get("ready") is not True or not citations_valid:
        return {"generated": False, "answer": None, "reason": "insufficient_grounding"}

    messages = build_citation_repair_messages(
        question, grounding_pack, generated_answer, validation_error
    )
    return {"generated": True, "answer": llm.generate(messages)}
