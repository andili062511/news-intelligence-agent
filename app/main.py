"""FastAPI entry point for the News Intelligence Agent."""

import os
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent.graph import run_agent
from llm.qwen import QwenLLM


app = FastAPI(
    title="News Intelligence Agent",
    description="Multi-source news intelligence agent based on RAG and tool orchestration.",
    version="0.1.0",
)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, examples=["What are the risks of rapidly advancing AI?"])


class Citation(BaseModel):
    evidence_id: str
    title: str
    source: str
    url: str
    published_at: str | None = None


class QueryResponse(BaseModel):
    question: str
    status: str
    answer: str | None
    citations: list[Citation]
    evidence_count: int
    intent: str
    search_queries: list[str]


@lru_cache(maxsize=1)
def get_llm() -> QwenLLM:
    """Load the answer model once, on the first query rather than app import."""

    return QwenLLM(
        model_name=os.getenv("NIA_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
        max_new_tokens=int(os.getenv("NIA_MAX_NEW_TOKENS", "256")),
    )


def _validated_citations(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Expose only evidence IDs accepted by final answer validation."""

    used_ids = set(state.get("answer_citations", []))
    if state.get("status") != "completed" or not used_ids:
        return []
    grounding_pack = state.get("grounding_pack") or {}
    return [
        {
            "evidence_id": str(item.get("evidence_id", "")),
            "title": str(item.get("title", "")),
            "source": str(item.get("source", "")),
            "url": str(item.get("url", "")),
            "published_at": (
                str(item["published_at"]) if item.get("published_at") else None
            ),
        }
        for item in grounding_pack.get("evidence", [])
        if item.get("evidence_id") in used_ids
    ]


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "News Intelligence Agent is running."}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query_news(payload: QueryRequest) -> QueryResponse:
    """Run the existing LangGraph workflow and return its validated result."""

    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="question must be non-empty")

    state = run_agent(question, llm=get_llm())
    completed = state.get("status") == "completed"
    answer = state.get("final_answer") if completed else None
    citations = _validated_citations(state) if completed else []
    return QueryResponse(
        question=question,
        status=str(state.get("status", "unknown")),
        answer=answer,
        citations=citations,
        evidence_count=len(state.get("evidence", [])),
        intent=str(state.get("intent", "")),
        search_queries=list(state.get("search_queries", [])),
    )
