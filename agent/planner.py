"""Deterministic planning and query rewriting for news questions."""

import re
from typing import Any


_COMPARE_PATTERN = re.compile(r"\b(compare|versus|vs\.?|difference)\b", re.IGNORECASE)
_TIME_PATTERN = re.compile(
    r"\b(latest|recent|recently|today|this week|current|newest)\b",
    re.IGNORECASE,
)


def _clean_question(question: str) -> str:
    cleaned = " ".join(question.strip().split())
    cleaned = re.sub(
        r"^(what|which|who|where|when|why|how)\s+"
        r"(are|is|were|was|do|does|did)\s+(?:the\s+)?",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.rstrip(" ?.!")


def plan_question(question: str) -> dict[str, Any]:
    """Classify a question and produce one or more offline search queries."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    cleaned = _clean_question(question)
    if _COMPARE_PATTERN.search(question):
        intent = "compare_news"
    elif _TIME_PATTERN.search(question):
        intent = "time_sensitive_news"
    else:
        intent = "news_search"

    queries = [cleaned]
    if intent == "time_sensitive_news":
        topic = _TIME_PATTERN.sub("", cleaned)
        topic = " ".join(topic.split()).strip(" -:,")
        recent_query = f"{topic} recent news" if topic else "recent news"
        if recent_query.casefold() != cleaned.casefold():
            queries.append(recent_query)

    return {
        "intent": intent,
        "search_queries": queries,
        "time_range": "this_week" if intent == "time_sensitive_news" else None,
    }


def rewrite_query(question: str, retry_count: int) -> str:
    """Broaden a query deterministically while keeping retries distinct."""

    base = _clean_question(question)
    replacements = (
        (r"\bAI\b", "artificial intelligence"),
        (r"\bmachines\b", "advanced AI"),
        (r"\bbecoming dangerously capable\b", "safety risks artificial intelligence capabilities"),
    )
    rewritten = base
    for pattern, replacement in replacements:
        rewritten = re.sub(pattern, replacement, rewritten, flags=re.IGNORECASE)
    rewritten = " ".join(rewritten.split())

    suffixes = ("news analysis risks developments", "reports research policy updates")
    attempt = max(1, retry_count)
    suffix = suffixes[(attempt - 1) % len(suffixes)]
    cycle = (attempt - 1) // len(suffixes)
    if rewritten.casefold() == base.casefold() or suffix.casefold() not in rewritten.casefold():
        rewritten = f"{rewritten} {suffix}"
    if cycle:
        rewritten = f"{rewritten} broader context {cycle + 1}"
    return rewritten
