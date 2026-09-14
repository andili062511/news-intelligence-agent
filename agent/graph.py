"""LangGraph planner-executor workflow for news evidence retrieval."""

from collections.abc import Callable
from typing import Any

from langgraph.graph import END, START, StateGraph
from llm.base import BaseLLM

from .evidence import assess_evidence
from .citations import (
    assign_evidence_ids,
    detect_duplicate_evidence,
)
from .grounding import assess_grounding_readiness, build_grounding_pack
from .generator import generate_grounded_answer, repair_grounded_answer
from .structured_answer import parse_claim_response, render_claims, validate_claims
from .planner import plan_question, rewrite_query
from .state import AgentState
from .tools import search_news_tool

SearchTool = Callable[[str, int, int], list[dict[str, Any]]]


def _rerank_score(item: dict[str, Any]) -> float:
    value = item.get("rerank_score")
    return float(value) if isinstance(value, (int, float)) else float("-inf")


def _merge_evidence(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    anonymous = 0
    for item in [*existing, *incoming]:
        key = str(item.get("chunk_id", "")).strip()
        if not key:
            anonymous += 1
            key = f"__anonymous_{anonymous}"
        previous = merged.get(key)
        if previous is None or _rerank_score(item) > _rerank_score(previous):
            merged[key] = dict(item)
    return sorted(
        merged.values(),
        key=_rerank_score,
        reverse=True,
    )


def create_agent_graph(
    search_tool: SearchTool = search_news_tool,
    *,
    llm: BaseLLM | None = None,
    top_k: int = 5,
    candidate_k: int = 15,
    min_evidence: int = 2,
    grounding_max_items: int = 3,
    max_chunks_per_article: int | None = 1,
):
    """Build and compile the agent graph with an injectable retrieval tool."""

    def planner_node(state: AgentState) -> dict[str, Any]:
        plan = plan_question(state["question"])
        return {
            **plan,
            "evidence": [],
            "missing_facts": [],
            "retry_count": state.get("retry_count", 0),
            "max_retries": state.get("max_retries", 2),
            "final_answer": None,
            "status": "planning",
            "reason": "",
            "evidence_reason": "",
            "grounding_pack": None,
            "citation_status": None,
            "citation_errors": [],
            "generated_answer": None,
            "raw_generated_answer": None,
            "raw_model_output": None,
            "structured_claims": [],
            "generation_attempts": 0,
            "generation_errors": [],
            "generation_validation_errors": [],
            "answer_citations": [],
            "answer_validation": None,
        }

    def search_node(state: AgentState) -> dict[str, Any]:
        found: list[dict[str, Any]] = []
        for query in state.get("search_queries", []):
            found.extend(search_tool(query, top_k, candidate_k))
        return {
            "evidence": _merge_evidence(state.get("evidence", []), found),
            "status": "searching",
        }

    def evidence_check_node(state: AgentState) -> dict[str, Any]:
        assessment = assess_evidence(state.get("evidence", []), min_items=min_evidence)
        return {
            "missing_facts": assessment["missing_facts"],
            "reason": assessment["reason"],
            "evidence_reason": assessment["reason"],
            "status": "verifying_citations" if assessment["sufficient"] else "checking_evidence",
        }

    def route_after_check(state: AgentState) -> str:
        if state.get("status") == "verifying_citations":
            return "citation_prepare"
        if state.get("retry_count", 0) < state.get("max_retries", 2):
            return "query_rewrite"
        return "insufficient"

    def query_rewrite_node(state: AgentState) -> dict[str, Any]:
        retry_count = state.get("retry_count", 0) + 1
        return {
            "search_queries": [rewrite_query(state["question"], retry_count)],
            "retry_count": retry_count,
            "status": "rewriting",
        }

    def citation_prepare_node(state: AgentState) -> dict[str, Any]:
        deduplicated, _ = detect_duplicate_evidence(state.get("evidence", []))
        identified = assign_evidence_ids(deduplicated)
        grounding_pack = build_grounding_pack(
            state["question"],
            state.get("evidence", []),
            max_items=grounding_max_items,
            max_chunks_per_article=max_chunks_per_article,
        )
        return {
            "evidence": identified,
            "grounding_pack": grounding_pack,
            "citation_errors": grounding_pack["citation_errors"],
            "citation_status": (
                "FAILED" if grounding_pack["citation_errors"] else "VALID"
            ),
            "status": "verifying_citations",
        }

    def citation_validate_node(state: AgentState) -> dict[str, Any]:
        if state.get("citation_errors"):
            return {
                "status": "citation_failed",
                "citation_status": "FAILED",
                "reason": "; ".join(
                    f"{error.get('evidence_id', 'unknown')}: {error.get('reason', 'invalid citation')}"
                    for error in state["citation_errors"]
                ),
                "final_answer": None,
            }

        readiness = assess_grounding_readiness(
            state.get("grounding_pack") or {}, min_evidence=min_evidence
        )
        grounding_pack = {
            **(state.get("grounding_pack") or {}),
            "ready": readiness["ready"],
            "readiness_reason": readiness["reason"],
        }
        if not readiness["ready"]:
            return {
                "status": "insufficient_evidence",
                "citation_status": "VALID",
                "reason": readiness["reason"],
                "grounding_pack": grounding_pack,
                "final_answer": None,
            }
        return {
            "status": "grounded",
            "citation_status": "VALID",
            "reason": readiness["reason"],
            "grounding_pack": grounding_pack,
            "final_answer": None,
        }

    def route_after_citation_validation(state: AgentState) -> str:
        if state.get("status") == "grounded":
            return "generate_answer"
        return "end"

    def generate_answer_node(state: AgentState) -> dict[str, Any]:
        if llm is None:
            raise RuntimeError("an llm must be provided when grounding is ready")
        result = generate_grounded_answer(
            state["question"], state.get("grounding_pack") or {}, llm
        )
        if not result["generated"]:
            return {
                "status": "insufficient_evidence",
                "reason": result["reason"],
                "generated_answer": None,
                "final_answer": None,
            }
        return {
            "status": "validating_answer_citations",
            "generated_answer": result["answer"],
            "raw_generated_answer": result["answer"],
            "raw_model_output": result["answer"],
            "generation_attempts": state.get("generation_attempts", 0) + 1,
            "final_answer": None,
        }

    def answer_citation_validate_node(state: AgentState) -> dict[str, Any]:
        parsed = parse_claim_response(state.get("raw_model_output") or "")
        validation = (
            validate_claims(parsed["claims"], state.get("grounding_pack") or {})
            if parsed["valid"]
            else parsed
        )
        if not validation["valid"]:
            errors = [*state.get("generation_validation_errors", []), validation]
            generation_errors = [
                *state.get("generation_errors", []), *validation["errors"]
            ]
            can_repair = state.get("generation_attempts", 0) < 2
            return {
                "status": "repairing_answer" if can_repair else "citation_failed",
                "structured_claims": validation["claims"],
                "answer_citations": [],
                "answer_validation": validation,
                "generation_validation_errors": errors,
                "generation_errors": generation_errors,
                "reason": "; ".join(validation["errors"]),
                "final_answer": None,
            }
        claims = validation["claims"]
        return {
            "status": "completed",
            "structured_claims": claims,
            "answer_citations": list(
                dict.fromkeys(citation for claim in claims for citation in claim["citations"])
            ),
            "answer_validation": validation,
            "reason": "valid",
            "final_answer": render_claims(claims),
        }

    def route_after_answer_validation(state: AgentState) -> str:
        return "repair_answer" if state.get("status") == "repairing_answer" else "end"

    def repair_answer_node(state: AgentState) -> dict[str, Any]:
        if llm is None:
            raise RuntimeError("an llm must be provided when grounding is ready")
        result = repair_grounded_answer(
            state["question"],
            state.get("grounding_pack") or {},
            state.get("raw_model_output") or "",
            state.get("reason") or "citation validation failed",
            llm,
        )
        if not result["generated"]:
            return {
                "status": "citation_failed",
                "reason": result["reason"],
                "final_answer": None,
            }
        return {
            "status": "validating_answer_citations",
            "generated_answer": result["answer"],
            "raw_generated_answer": result["answer"],
            "raw_model_output": result["answer"],
            "generation_attempts": state.get("generation_attempts", 0) + 1,
            "final_answer": None,
        }

    def insufficient_node(state: AgentState) -> dict[str, Any]:
        return {"status": "insufficient_evidence", "final_answer": None}

    builder = StateGraph(AgentState)
    builder.add_node("planner", planner_node)
    builder.add_node("search", search_node)
    builder.add_node("evidence_check", evidence_check_node)
    builder.add_node("query_rewrite", query_rewrite_node)
    builder.add_node("citation_prepare", citation_prepare_node)
    builder.add_node("citation_validate", citation_validate_node)
    builder.add_node("generation", generate_answer_node)
    builder.add_node("answer_validation", answer_citation_validate_node)
    builder.add_node("citation_repair", repair_answer_node)
    builder.add_node("insufficient", insufficient_node)
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "search")
    builder.add_edge("search", "evidence_check")
    builder.add_conditional_edges(
        "evidence_check",
        route_after_check,
        {
            "citation_prepare": "citation_prepare",
            "query_rewrite": "query_rewrite",
            "insufficient": "insufficient",
        },
    )
    builder.add_edge("query_rewrite", "search")
    builder.add_edge("citation_prepare", "citation_validate")
    builder.add_conditional_edges(
        "citation_validate",
        route_after_citation_validation,
        {"generate_answer": "generation", "end": END},
    )
    builder.add_edge("generation", "answer_validation")
    builder.add_conditional_edges(
        "answer_validation",
        route_after_answer_validation,
        {"repair_answer": "citation_repair", "end": END},
    )
    builder.add_edge("citation_repair", "answer_validation")
    builder.add_edge("insufficient", END)
    return builder.compile()


build_agent_graph = create_agent_graph


def run_agent(
    question: str,
    *,
    llm: BaseLLM | None = None,
    max_retries: int = 2,
    search_tool: SearchTool = search_news_tool,
    top_k: int = 5,
    candidate_k: int = 15,
    grounding_max_items: int = 3,
    max_chunks_per_article: int | None = 1,
) -> AgentState:
    """Run the compiled workflow and return its structured final state."""

    graph = create_agent_graph(
        search_tool,
        llm=llm,
        top_k=top_k,
        candidate_k=candidate_k,
        grounding_max_items=grounding_max_items,
        max_chunks_per_article=max_chunks_per_article,
    )
    return graph.invoke({"question": question, "max_retries": max(0, max_retries)})
