"""Validate the generated planner train and validation datasets."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from finetune.schemas import DatasetValidationError, PlannerOutput, validate_sample


@dataclass(frozen=True, slots=True)
class ValidationResult:
    sample_count: int
    intent_counts: Counter[str]
    instructions: frozenset[str]
    duplicate_count: int


def validate_jsonl(path: Path | str) -> ValidationResult:
    """Validate every JSONL row and report intent and duplicate statistics."""

    dataset_path = Path(path)
    counts: Counter[str] = Counter()
    instructions: set[str] = set()
    duplicates = 0
    sample_count = 0

    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise DatasetValidationError(f"{dataset_path}:{line_number}: blank JSONL line")
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetValidationError(
                    f"{dataset_path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            try:
                instruction, output = validate_sample(sample)
            except DatasetValidationError as exc:
                raise DatasetValidationError(f"{dataset_path}:{line_number}: {exc}") from exc
            sample_count += 1
            counts[output.intent] += 1
            normalized = instruction.casefold()
            if normalized in instructions:
                duplicates += 1
            instructions.add(normalized)

    if sample_count == 0:
        raise DatasetValidationError(f"{dataset_path}: dataset is empty")
    return ValidationResult(sample_count, counts, frozenset(instructions), duplicates)


def validate_datasets(train_path: Path | str, validation_path: Path | str) -> tuple[ValidationResult, ValidationResult, int]:
    train = validate_jsonl(train_path)
    validation = validate_jsonl(validation_path)
    overlap = len(train.instructions & validation.instructions)
    if train.duplicate_count or validation.duplicate_count:
        raise DatasetValidationError(
            f"exact duplicate instructions found: {train.duplicate_count + validation.duplicate_count}"
        )
    if overlap:
        raise DatasetValidationError(f"train/validation instruction overlap found: {overlap}")
    return train, validation, overlap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_dir = Path(__file__).parent / "data"
    parser.add_argument("--train", type=Path, default=default_dir / "planner_train.jsonl")
    parser.add_argument("--validation", type=Path, default=default_dir / "planner_val.jsonl")
    args = parser.parse_args()

    try:
        train, validation, overlap = validate_datasets(args.train, args.validation)
    except (DatasetValidationError, OSError) as exc:
        print(f"Dataset validation: FAILED\n{exc}")
        raise SystemExit(1) from exc

    combined = train.intent_counts + validation.intent_counts
    print(f"Train samples: {train.sample_count}")
    print(f"Validation samples: {validation.sample_count}")
    print("\nIntent distribution:")
    for intent in ("news_search", "compare_news", "time_sensitive_news", "fact_check"):
        print(f"{intent}: {combined[intent]}")
    print(f"\nDuplicate instructions: {train.duplicate_count + validation.duplicate_count}")
    print(f"Train/Val overlap: {overlap}")
    print("\nDataset validation: PASSED")


if __name__ == "__main__":
    main()
