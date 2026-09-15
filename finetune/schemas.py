"""Schemas and validation rules for planner instruction samples."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Literal, TypedDict, cast


Intent = Literal[
    "news_search",
    "compare_news",
    "time_sensitive_news",
    "fact_check",
]
Tool = Literal[
    "news_search",
    "time_filter",
    "source_compare",
    "structured_query",
]

ALLOWED_INTENTS = frozenset(
    {"news_search", "compare_news", "time_sensitive_news", "fact_check"}
)
ALLOWED_TOOLS = frozenset(
    {"news_search", "time_filter", "source_compare", "structured_query"}
)


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class InstructionSample(TypedDict):
    messages: list[Message]


class DatasetValidationError(ValueError):
    """Raised when a planner sample violates the dataset contract."""


@dataclass(frozen=True, slots=True)
class PlannerOutput:
    """Structured action plan learned by the fine-tuned planner."""

    intent: Intent
    search_queries: list[str]
    time_range: str | None
    tools: list[Tool]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))


def validate_planner_output(value: object) -> PlannerOutput:
    """Validate a decoded assistant JSON object and return its typed form."""

    if not isinstance(value, dict):
        raise DatasetValidationError("assistant content must decode to a JSON object")

    required = {"intent", "search_queries", "time_range", "tools"}
    if set(value) != required:
        missing = sorted(required - set(value))
        extra = sorted(set(value) - required)
        raise DatasetValidationError(f"planner output fields invalid; missing={missing}, extra={extra}")

    intent = value["intent"]
    if not isinstance(intent, str) or intent not in ALLOWED_INTENTS:
        raise DatasetValidationError(f"invalid intent: {intent!r}")

    queries = value["search_queries"]
    if (
        not isinstance(queries, list)
        or not queries
        or any(not isinstance(query, str) or not query.strip() for query in queries)
    ):
        raise DatasetValidationError("search_queries must be a non-empty list of strings")
    if intent == "compare_news" and len(queries) < 2:
        raise DatasetValidationError("compare_news requires at least two search queries")

    time_range = value["time_range"]
    if time_range is not None and (not isinstance(time_range, str) or not time_range.strip()):
        raise DatasetValidationError("time_range must be null or a non-empty string")
    if intent == "time_sensitive_news" and time_range is None:
        raise DatasetValidationError("time_sensitive_news requires time_range")

    tools = value["tools"]
    if (
        not isinstance(tools, list)
        or not tools
        or any(not isinstance(tool, str) or tool not in ALLOWED_TOOLS for tool in tools)
    ):
        raise DatasetValidationError("tools must be a non-empty list containing only allowed tools")
    if len(tools) != len(set(tools)):
        raise DatasetValidationError("tools must not contain duplicates")

    return PlannerOutput(
        intent=cast(Intent, intent),
        search_queries=list(queries),
        time_range=time_range,
        tools=cast(list[Tool], list(tools)),
    )


def validate_sample(value: object) -> tuple[str, PlannerOutput]:
    """Validate one messages-format training sample.

    Returns the normalized user instruction and parsed planner output.
    """

    if not isinstance(value, dict) or set(value) != {"messages"}:
        raise DatasetValidationError("sample must contain exactly one 'messages' field")
    messages = value["messages"]
    if not isinstance(messages, list) or len(messages) != 3:
        raise DatasetValidationError("messages must contain system, user, and assistant entries")

    expected_roles = ["system", "user", "assistant"]
    for index, (message, expected_role) in enumerate(zip(messages, expected_roles)):
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise DatasetValidationError(f"message {index} must contain role and content")
        if message["role"] != expected_role:
            raise DatasetValidationError(
                f"message roles must be {expected_roles}, got {message['role']!r} at index {index}"
            )
        if not isinstance(message["content"], str) or not message["content"].strip():
            raise DatasetValidationError(f"message {index} content must be a non-empty string")

    try:
        output = json.loads(messages[2]["content"])
    except json.JSONDecodeError as exc:
        raise DatasetValidationError("assistant content must be valid JSON") from exc

    instruction = " ".join(messages[1]["content"].split())
    return instruction, validate_planner_output(output)
