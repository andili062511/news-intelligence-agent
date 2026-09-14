"""Shared state for the planner-executor news agent."""

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """State passed between LangGraph nodes.

    ``total=False`` lets callers start a run with only a question; the planner
    node fills in all runtime defaults deterministically.
    """

    question: str
    intent: str
    search_queries: list[str]
    time_range: str | None
    evidence: list[dict[str, Any]]
    missing_facts: list[str]
    retry_count: int
    max_retries: int
    final_answer: str | None
    status: str
    reason: str
    evidence_reason: str
    grounding_pack: dict[str, Any] | None
    citation_status: str | None
    citation_errors: list[dict[str, Any]]
    generated_answer: str | None
    raw_generated_answer: str | None
    raw_model_output: str | None
    structured_claims: list[dict[str, Any]]
    generation_attempts: int
    generation_errors: list[str]
    generation_validation_errors: list[dict[str, Any]]
    answer_citations: list[str]
    answer_validation: dict[str, Any] | None
