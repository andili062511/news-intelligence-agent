"""Evaluate planner behavior on the validation conversational dataset."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from finetune.inference_adapter import PlannerInference
from finetune.schemas import DatasetValidationError, PlannerOutput, validate_planner_output, validate_sample
from finetune.training_utils import load_training_config


@dataclass(frozen=True, slots=True)
class PlannerMetrics:
    sample_count: int
    intent_accuracy: float
    tool_exact_match: float
    time_range_accuracy: float
    json_valid_rate: float
    schema_valid_rate: float
    search_query_nonempty_rate: float
    full_exact_match: float

    def as_dict(self) -> dict[str, int | float]:
        return {
            "sample_count": self.sample_count,
            "intent_accuracy": self.intent_accuracy,
            "tool_exact_match": self.tool_exact_match,
            "time_range_accuracy": self.time_range_accuracy,
            "json_valid_rate": self.json_valid_rate,
            "schema_valid_rate": self.schema_valid_rate,
            "search_query_nonempty_rate": self.search_query_nonempty_rate,
            "full_exact_match": self.full_exact_match,
        }


def _expected_dict(value: PlannerOutput | dict[str, object]) -> dict[str, object]:
    if isinstance(value, PlannerOutput):
        return value.to_dict()
    return validate_planner_output(value).to_dict()


def score_predictions(
    raw_predictions: Sequence[str],
    expected_outputs: Sequence[PlannerOutput | dict[str, object]],
) -> PlannerMetrics:
    """Score raw generations; malformed predictions receive zero for field metrics."""

    if len(raw_predictions) != len(expected_outputs):
        raise ValueError("predictions and expected outputs must have equal length")
    if not raw_predictions:
        raise ValueError("at least one prediction is required")

    counts = {
        "intent": 0,
        "tools": 0,
        "time_range": 0,
        "json": 0,
        "schema": 0,
        "queries": 0,
        "full": 0,
    }
    for raw, expected_value in zip(raw_predictions, expected_outputs):
        expected = _expected_dict(expected_value)
        try:
            decoded = json.loads(raw.strip())
        except (json.JSONDecodeError, AttributeError):
            continue
        counts["json"] += 1
        if not isinstance(decoded, dict):
            continue

        queries = decoded.get("search_queries")
        if (
            isinstance(queries, list)
            and bool(queries)
            and all(isinstance(query, str) and query.strip() for query in queries)
        ):
            counts["queries"] += 1
        if decoded.get("intent") == expected["intent"]:
            counts["intent"] += 1
        if decoded.get("tools") == expected["tools"]:
            counts["tools"] += 1
        if decoded.get("time_range") == expected["time_range"]:
            counts["time_range"] += 1
        try:
            normalized = validate_planner_output(decoded).to_dict()
        except DatasetValidationError:
            continue
        counts["schema"] += 1
        if normalized == expected:
            counts["full"] += 1

    total = len(raw_predictions)
    return PlannerMetrics(
        sample_count=total,
        intent_accuracy=counts["intent"] / total,
        tool_exact_match=counts["tools"] / total,
        time_range_accuracy=counts["time_range"] / total,
        json_valid_rate=counts["json"] / total,
        schema_valid_rate=counts["schema"] / total,
        search_query_nonempty_rate=counts["queries"] / total,
        full_exact_match=counts["full"] / total,
    )


def load_evaluation_samples(path: Path | str) -> tuple[list[str], list[PlannerOutput]]:
    """Load validation prompts and gold planner objects through HF Datasets."""

    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    from datasets import load_dataset

    dataset = load_dataset("json", data_files={"validation": str(dataset_path)})[
        "validation"
    ]
    questions: list[str] = []
    outputs: list[PlannerOutput] = []
    for row_number, sample in enumerate(dataset, start=1):
        try:
            question, output = validate_sample(sample)
        except DatasetValidationError as exc:
            raise ValueError(f"validation row {row_number}: {exc}") from exc
        questions.append(question)
        outputs.append(output)
    if not questions:
        raise ValueError("validation dataset is empty")
    return questions, outputs


def select_evaluation_samples(
    questions: Sequence[str],
    expected_outputs: Sequence[PlannerOutput],
    max_samples: int | None,
) -> tuple[list[str], list[PlannerOutput]]:
    """Select the same deterministic prefix of prompts and expected outputs."""

    if len(questions) != len(expected_outputs):
        raise ValueError("questions and expected outputs must have equal length")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be a positive integer or None")
    limit = len(questions) if max_samples is None else min(max_samples, len(questions))
    return list(questions[:limit]), list(expected_outputs[:limit])


def evaluate(
    planner: PlannerInference,
    questions: Iterable[str],
    expected_outputs: Sequence[PlannerOutput],
    *,
    max_new_tokens: int = 256,
) -> PlannerMetrics:
    predictions = [
        planner.generate(question, max_new_tokens=max_new_tokens).raw_response
        for question in questions
    ]
    return score_predictions(predictions, expected_outputs)


def print_metrics(metrics: PlannerMetrics) -> None:
    print(f"Evaluated samples: {metrics.sample_count}")
    print(f"Intent Accuracy: {metrics.intent_accuracy:.2f}")
    print(f"Tool Exact Match: {metrics.tool_exact_match:.2f}")
    print(f"Time Range Accuracy: {metrics.time_range_accuracy:.2f}")
    print(f"JSON Valid Rate: {metrics.json_valid_rate:.2f}")
    print(f"Schema Valid Rate: {metrics.schema_valid_rate:.2f}")
    print(f"Search Query Non-empty Rate: {metrics.search_query_nonempty_rate:.2f}")
    print(f"Full Exact Match: {metrics.full_exact_match:.2f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=load_training_config().model_name)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument(
        "--val-file", type=Path, default=Path("finetune/data/planner_val.jsonl")
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--max-samples",
        type=_positive_int,
        default=None,
        help="evaluate the first N validation samples (default: all)",
    )
    parser.add_argument(
        "--load-in-4bit", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main() -> None:
    args = build_parser().parse_args()
    questions, expected = load_evaluation_samples(args.val_file)
    questions, expected = select_evaluation_samples(
        questions, expected, args.max_samples
    )
    planner = PlannerInference(
        args.model, args.adapter, load_in_4bit=args.load_in_4bit
    )
    metrics = evaluate(
        planner, questions, expected, max_new_tokens=args.max_new_tokens
    )
    label = "base + adapter" if args.adapter else "base model"
    print(f"Evaluation target: {label}")
    print_metrics(metrics)


if __name__ == "__main__":
    main()
