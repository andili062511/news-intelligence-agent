"""Evaluate planner behavior on any messages-format evaluation JSONL dataset."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from finetune.inference_adapter import PlannerInference
from finetune.schemas import (
    ALLOWED_INTENTS,
    DatasetValidationError,
    PlannerOutput,
    validate_planner_output,
    validate_sample,
)
from finetune.training_utils import load_training_config


INTENTS = ("news_search", "compare_news", "time_sensitive_news", "fact_check")
QUERY_STOPWORDS = frozenset(
    {"the", "a", "an", "is", "are", "what", "how", "about", "latest", "recent"}
)
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*")


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
    structural_exact_match: float
    tool_precision: float
    tool_recall: float
    tool_f1: float
    query_count_accuracy: float
    entity_coverage: float
    query_token_recall: float

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
            "structural_exact_match": self.structural_exact_match,
            "tool_precision": self.tool_precision,
            "tool_recall": self.tool_recall,
            "tool_f1": self.tool_f1,
            "query_count_accuracy": self.query_count_accuracy,
            "entity_coverage": self.entity_coverage,
            "query_token_recall": self.query_token_recall,
        }


@dataclass(frozen=True, slots=True)
class PredictionAnalysis:
    expected: dict[str, object]
    predicted: dict[str, object] | None
    raw_model_response: str
    errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    metrics: PlannerMetrics
    analyses: tuple[PredictionAnalysis, ...]
    intent_confusion_matrix: dict[str, dict[str, int]]
    per_intent_metrics: dict[str, dict[str, int | float]]
    invalid_intent_prediction_count: int
    common_time_range_errors: tuple[dict[str, object], ...]

    def summary_dict(self) -> dict[str, object]:
        summary: dict[str, object] = self.metrics.as_dict()
        summary.update(
            {
                "tool_metric_averaging": "micro over tool-set membership decisions",
                "intent_confusion_matrix": self.intent_confusion_matrix,
                "per_intent_metrics": self.per_intent_metrics,
                "invalid_intent_prediction_count": self.invalid_intent_prediction_count,
                "common_time_range_errors": list(self.common_time_range_errors),
                "query_metric_method": {
                    "normalization": "lowercase alphanumeric tokens with fixed stopwords removed",
                    "entity_detection": (
                        "capitalized expected-query tokens (titlecase, acronym, or internal "
                        "capitals), excluding fixed stopwords"
                    ),
                    "aggregation": "micro token/entity recall; query-count accuracy per sample",
                },
            }
        )
        return summary


def _expected_dict(value: PlannerOutput | dict[str, object]) -> dict[str, object]:
    if isinstance(value, PlannerOutput):
        return value.to_dict()
    return validate_planner_output(value).to_dict()


def normalize_query_tokens(queries: object) -> set[str]:
    """Return deterministic, case-folded query tokens with fixed stopwords removed."""

    if not isinstance(queries, list):
        return set()
    return {
        token.lower()
        for query in queries
        if isinstance(query, str)
        for token in TOKEN_PATTERN.findall(query)
        if token.lower() not in QUERY_STOPWORDS
    }


def extract_entity_tokens(queries: object) -> set[str]:
    """Extract obvious proper-name tokens from expected queries without an NLP model.

    Internal capitals (``OpenAI``), acronyms, and titlecase tokens are treated as
    entities. This deliberately favors transparent recall over linguistic inference.
    """

    if not isinstance(queries, list):
        return set()
    entities: set[str] = set()
    for query in queries:
        if not isinstance(query, str):
            continue
        tokens = TOKEN_PATTERN.findall(query)
        for token in tokens:
            if token.lower() in QUERY_STOPWORDS:
                continue
            has_internal_upper = any(character.isupper() for character in token[1:])
            is_acronym = len(token) > 1 and token.isupper()
            is_titlecase = token[:1].isupper()
            if has_internal_upper or is_acronym or is_titlecase:
                entities.add(token.lower())
    return entities


def _safe_divide(numerator: int | float, denominator: int | float, *, empty: float = 0.0) -> float:
    return numerator / denominator if denominator else empty


def analyze_predictions(
    raw_predictions: Sequence[str],
    expected_outputs: Sequence[PlannerOutput | dict[str, object]],
) -> EvaluationResult:
    """Compute aggregate metrics and deterministic per-sample error analysis."""

    if len(raw_predictions) != len(expected_outputs):
        raise ValueError("predictions and expected outputs must have equal length")
    if not raw_predictions:
        raise ValueError("at least one prediction is required")

    counts = Counter()
    tool_tp = tool_fp = tool_fn = 0
    expected_token_total = matched_token_total = 0
    expected_entity_total = matched_entity_total = 0
    confusion = {expected: {predicted: 0 for predicted in INTENTS} for expected in INTENTS}
    intent_support = Counter()
    intent_predicted = Counter()
    intent_tp = Counter()
    invalid_intents = 0
    time_errors: Counter[tuple[str | None, str | None]] = Counter()
    analyses: list[PredictionAnalysis] = []

    for raw, expected_value in zip(raw_predictions, expected_outputs):
        expected = _expected_dict(expected_value)
        errors: list[str] = []
        decoded: dict[str, object] | None = None
        schema_valid = False
        try:
            candidate = json.loads(raw.strip())
        except (json.JSONDecodeError, AttributeError):
            errors.append("invalid_json")
        else:
            counts["json"] += 1
            if isinstance(candidate, dict):
                decoded = candidate
                try:
                    validate_planner_output(decoded)
                except DatasetValidationError:
                    errors.append("schema_invalid")
                else:
                    schema_valid = True
                    counts["schema"] += 1
            else:
                errors.append("schema_invalid")

        predicted = decoded or {}
        expected_intent = str(expected["intent"])
        predicted_intent = predicted.get("intent")
        intent_support[expected_intent] += 1
        if isinstance(predicted_intent, str) and predicted_intent in ALLOWED_INTENTS:
            predicted_intent = str(predicted_intent)
            confusion[expected_intent][predicted_intent] += 1
            intent_predicted[predicted_intent] += 1
            if predicted_intent == expected_intent:
                intent_tp[expected_intent] += 1
        else:
            invalid_intents += 1

        if predicted_intent == expected["intent"]:
            counts["intent"] += 1
        else:
            errors.append("intent_mismatch")

        expected_tools_list = expected["tools"]
        predicted_tools_value = predicted.get("tools")
        if predicted_tools_value == expected_tools_list:
            counts["tools"] += 1
        else:
            errors.append("tool_mismatch")
        expected_tools = set(expected_tools_list) if isinstance(expected_tools_list, list) else set()
        predicted_tools = (
            {tool for tool in predicted_tools_value if isinstance(tool, str)}
            if isinstance(predicted_tools_value, list)
            else set()
        )
        tool_tp += len(expected_tools & predicted_tools)
        tool_fp += len(predicted_tools - expected_tools)
        tool_fn += len(expected_tools - predicted_tools)

        expected_time = expected["time_range"]
        predicted_time = predicted.get("time_range")
        if predicted_time == expected_time:
            counts["time_range"] += 1
        else:
            errors.append("time_range_mismatch")
            normalized_predicted_time = predicted_time if isinstance(predicted_time, str) else None
            time_errors[(expected_time, normalized_predicted_time)] += 1

        expected_queries = expected["search_queries"]
        predicted_queries = predicted.get("search_queries")
        valid_nonempty_queries = (
            isinstance(predicted_queries, list)
            and bool(predicted_queries)
            and all(isinstance(query, str) and query.strip() for query in predicted_queries)
        )
        if valid_nonempty_queries:
            counts["queries"] += 1
        if isinstance(predicted_queries, list) and len(predicted_queries) == len(expected_queries):
            counts["query_count"] += 1
        else:
            errors.append("query_count_mismatch")

        expected_tokens = normalize_query_tokens(expected_queries)
        predicted_tokens = normalize_query_tokens(predicted_queries)
        token_matches = expected_tokens & predicted_tokens
        expected_token_total += len(expected_tokens)
        matched_token_total += len(token_matches)
        if token_matches != expected_tokens:
            errors.append("query_token_recall_incomplete")

        expected_entities = extract_entity_tokens(expected_queries)
        entity_matches = expected_entities & predicted_tokens
        expected_entity_total += len(expected_entities)
        matched_entity_total += len(entity_matches)
        if entity_matches != expected_entities:
            errors.append("entity_coverage_incomplete")

        structural_match = (
            predicted_intent == expected["intent"]
            and predicted_time == expected_time
            and predicted_tools_value == expected_tools_list
        )
        if structural_match:
            counts["structural"] += 1

        if schema_valid and validate_planner_output(predicted).to_dict() == expected:
            counts["full"] += 1
        else:
            errors.append("full_exact_mismatch")

        analyses.append(
            PredictionAnalysis(
                expected=expected,
                predicted=decoded,
                raw_model_response=raw,
                errors=tuple(dict.fromkeys(errors)),
            )
        )

    total = len(raw_predictions)
    precision = _safe_divide(tool_tp, tool_tp + tool_fp)
    recall = _safe_divide(tool_tp, tool_tp + tool_fn)
    tool_f1 = _safe_divide(2 * precision * recall, precision + recall)
    metrics = PlannerMetrics(
        sample_count=total,
        intent_accuracy=counts["intent"] / total,
        tool_exact_match=counts["tools"] / total,
        time_range_accuracy=counts["time_range"] / total,
        json_valid_rate=counts["json"] / total,
        schema_valid_rate=counts["schema"] / total,
        search_query_nonempty_rate=counts["queries"] / total,
        full_exact_match=counts["full"] / total,
        structural_exact_match=counts["structural"] / total,
        tool_precision=precision,
        tool_recall=recall,
        tool_f1=tool_f1,
        query_count_accuracy=counts["query_count"] / total,
        entity_coverage=_safe_divide(matched_entity_total, expected_entity_total, empty=1.0),
        query_token_recall=_safe_divide(matched_token_total, expected_token_total, empty=1.0),
    )

    per_intent: dict[str, dict[str, int | float]] = {}
    for intent in INTENTS:
        true_positive = intent_tp[intent]
        intent_precision = _safe_divide(true_positive, intent_predicted[intent])
        intent_recall = _safe_divide(true_positive, intent_support[intent])
        per_intent[intent] = {
            "precision": intent_precision,
            "recall": intent_recall,
            "f1": _safe_divide(2 * intent_precision * intent_recall, intent_precision + intent_recall),
            "support": intent_support[intent],
        }

    common_time_errors = tuple(
        {"expected": expected, "predicted": predicted, "count": count}
        for (expected, predicted), count in sorted(
            time_errors.items(), key=lambda item: (-item[1], str(item[0]))
        )
    )
    return EvaluationResult(
        metrics=metrics,
        analyses=tuple(analyses),
        intent_confusion_matrix=confusion,
        per_intent_metrics=per_intent,
        invalid_intent_prediction_count=invalid_intents,
        common_time_range_errors=common_time_errors,
    )


def score_predictions(
    raw_predictions: Sequence[str],
    expected_outputs: Sequence[PlannerOutput | dict[str, object]],
) -> PlannerMetrics:
    """Score raw generations; malformed predictions receive zero for field metrics."""

    return analyze_predictions(raw_predictions, expected_outputs).metrics


def load_evaluation_samples(path: Path | str) -> tuple[list[str], list[PlannerOutput]]:
    """Load prompts and gold planner objects from a messages-format JSONL file."""

    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    from datasets import load_dataset

    dataset = load_dataset("json", data_files={"validation": str(dataset_path)})["validation"]
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


def generate_predictions(
    planner: PlannerInference, questions: Iterable[str], *, max_new_tokens: int = 256
) -> list[str]:
    return [
        planner.generate(question, max_new_tokens=max_new_tokens).raw_response
        for question in questions
    ]


def evaluate(
    planner: PlannerInference,
    questions: Iterable[str],
    expected_outputs: Sequence[PlannerOutput],
    *,
    max_new_tokens: int = 256,
) -> PlannerMetrics:
    predictions = generate_predictions(planner, questions, max_new_tokens=max_new_tokens)
    return score_predictions(predictions, expected_outputs)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_error_report(
    path: Path | str, questions: Sequence[str], result: EvaluationResult
) -> None:
    """Write non-exact samples as JSONL, including the untouched raw response."""

    if len(questions) != len(result.analyses):
        raise ValueError("questions and analyses must have equal length")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for question, analysis in zip(questions, result.analyses):
        if not analysis.errors:
            continue
        rows.append(
            json.dumps(
                {
                    "question": question,
                    "expected": analysis.expected,
                    "predicted": analysis.predicted,
                    "errors": list(analysis.errors),
                    "raw_model_response": analysis.raw_model_response,
                },
                ensure_ascii=False,
            )
        )
    output_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def write_summary_report(path: Path | str, result: EvaluationResult) -> None:
    _write_json(Path(path), result.summary_dict())


def print_metrics(result: EvaluationResult | PlannerMetrics) -> None:
    metrics = result.metrics if isinstance(result, EvaluationResult) else result
    print(f"Evaluated samples: {metrics.sample_count}")
    print(f"Intent Accuracy: {metrics.intent_accuracy:.2f}")
    print(f"Tool Exact Match: {metrics.tool_exact_match:.2f}")
    print(f"Time Range Accuracy: {metrics.time_range_accuracy:.2f}")
    print(f"JSON Valid Rate: {metrics.json_valid_rate:.2f}")
    print(f"Schema Valid Rate: {metrics.schema_valid_rate:.2f}")
    print(f"Search Query Non-empty Rate: {metrics.search_query_nonempty_rate:.2f}")
    print(f"Full Exact Match: {metrics.full_exact_match:.2f}")
    print(f"Structural Exact Match: {metrics.structural_exact_match:.2f}")
    print(f"Tool Precision (micro): {metrics.tool_precision:.2f}")
    print(f"Tool Recall (micro): {metrics.tool_recall:.2f}")
    print(f"Tool F1 (micro): {metrics.tool_f1:.2f}")
    print(f"Query Count Accuracy: {metrics.query_count_accuracy:.2f}")
    print(f"Entity Coverage (micro): {metrics.entity_coverage:.2f}")
    print(f"Query Token Recall (micro): {metrics.query_token_recall:.2f}")
    if isinstance(result, EvaluationResult):
        print("Intent Confusion Matrix (Expected -> Predicted):")
        for expected in INTENTS:
            for predicted in INTENTS:
                count = result.intent_confusion_matrix[expected][predicted]
                if count:
                    print(f"  {expected} -> {predicted}: {count}")
        print("Per-intent metrics:")
        for intent, values in result.per_intent_metrics.items():
            print(
                f"  {intent}: precision={values['precision']:.2f}, "
                f"recall={values['recall']:.2f}, F1={values['f1']:.2f}, "
                f"support={values['support']}"
            )
        print("Common time-range errors:")
        for error in result.common_time_range_errors:
            print(f"  {error['expected']} -> {error['predicted']}: {error['count']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=load_training_config().model_name)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument(
        "--val-file",
        type=Path,
        default=Path("finetune/data/planner_val.jsonl"),
        help=(
            "messages-format evaluation JSONL (for example, "
            "finetune/data/planner_hard_eval.jsonl)"
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--max-samples",
        type=_positive_int,
        default=None,
        help="evaluate the first N evaluation samples (default: all)",
    )
    parser.add_argument("--error-output", type=Path)
    parser.add_argument("--summary-output", type=Path)
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
    questions, expected = select_evaluation_samples(questions, expected, args.max_samples)
    planner = PlannerInference(args.model, args.adapter, load_in_4bit=args.load_in_4bit)
    predictions = generate_predictions(planner, questions, max_new_tokens=args.max_new_tokens)
    result = analyze_predictions(predictions, expected)
    label = "base + adapter" if args.adapter else "base model"
    print(f"Evaluation target: {label}")
    print_metrics(result)
    if args.error_output:
        write_error_report(args.error_output, questions, result)
        print(f"Error report: {args.error_output}")
    if args.summary_output:
        write_summary_report(args.summary_output, result)
        print(f"Summary report: {args.summary_output}")


if __name__ == "__main__":
    main()
