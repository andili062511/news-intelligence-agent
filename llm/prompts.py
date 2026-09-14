"""Prompts for evidence-only news answer generation."""

from collections.abc import Mapping
from typing import Any


SYSTEM_PROMPT = """You are a news intelligence analyst.

You must return valid JSON only.

Schema:

{
  "claims": [
    {
      "text": "string",
      "citations": ["E1"]
    }
  ]
}

Rules:

1. Every claim must contain at least one citation.
2. Citation IDs must come only from the provided evidence.
3. Do not use outside knowledge.
4. If a claim cannot be supported by evidence, omit it.
5. Do not create E3/E4 etc unless they exist.
6. Do not include markdown.
7. Do not include explanatory text outside the JSON.
8. Prefer 2-4 concise claims."""


def build_grounded_messages(
    question: str,
    grounding_pack: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Build chat messages using only the pack's validated evidence projection."""

    sections = [f"Question:\n{question}", "Evidence:"]
    raw_evidence = grounding_pack.get("evidence", [])
    evidence = raw_evidence if isinstance(raw_evidence, list) else []
    for item in evidence:
        if not isinstance(item, Mapping):
            continue
        sections.append(
            "\n".join(
                (
                    f"[{item.get('evidence_id', '')}]",
                    f"Title: {item.get('title', '')}",
                    f"Source: {item.get('source', '')}",
                    f"Date: {item.get('published_at', '')}",
                    f"Text: {item.get('text', '')}",
                )
            )
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(sections)},
    ]


def build_citation_repair_messages(
    question: str,
    grounding_pack: Mapping[str, Any],
    generated_answer: str,
    validation_error: str,
) -> list[dict[str, str]]:
    """Build a one-shot repair request using the unchanged grounding pack."""

    grounded = build_grounded_messages(question, grounding_pack)
    repair = "\n\n".join(
        (
            f"Original question:\n{question}",
            f"Same grounding pack:\n{grounded[1]['content']}",
            f"First raw model output:\n{generated_answer}",
            f"Validation errors:\n{validation_error}",
            (
                "Generate the complete JSON object again. Follow the schema and all rules. "
                "Do not patch or explain the first output."
            ),
        )
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": repair}]
