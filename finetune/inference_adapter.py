"""Run deterministic planner inference with a base model and optional PEFT adapter."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from finetune.dataset_builder import SYSTEM_PROMPT
from finetune.generation_utils import generate_chat_completion
from finetune.schemas import DatasetValidationError, PlannerOutput, validate_planner_output
from finetune.training_utils import (
    DEFAULT_CONFIG_PATH,
    build_quantization_config,
    get_compute_dtype,
    load_training_config,
)


@dataclass(frozen=True, slots=True)
class InferenceResult:
    raw_response: str
    parsed_response: dict[str, object] | None
    schema_valid: bool
    error: str | None = None


def parse_planner_response(raw_response: str) -> InferenceResult:
    """Parse a strict JSON response and validate the planner schema."""

    try:
        decoded = json.loads(raw_response.strip())
    except json.JSONDecodeError as exc:
        return InferenceResult(raw_response, None, False, f"invalid JSON: {exc.msg}")
    if not isinstance(decoded, dict):
        return InferenceResult(raw_response, None, False, "response is not a JSON object")
    try:
        validated: PlannerOutput = validate_planner_output(decoded)
    except DatasetValidationError as exc:
        return InferenceResult(raw_response, decoded, False, str(exc))
    return InferenceResult(raw_response, validated.to_dict(), True)


class PlannerInference:
    """One loaded base model, optionally augmented with a PEFT adapter."""

    def __init__(
        self,
        model_name: str,
        adapter_path: Path | str | None = None,
        *,
        load_in_4bit: bool = True,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_options: dict[str, Any] = {"dtype": get_compute_dtype(torch)}
        if load_in_4bit:
            config = load_training_config(DEFAULT_CONFIG_PATH)
            model_options.update(
                quantization_config=build_quantization_config(
                    config.quantization, torch_module=torch
                ),
                device_map="auto",
            )
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_options)
        if adapter_path is not None:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, str(adapter_path))
        self.model.eval()

    def generate(self, question: str, *, max_new_tokens: int = 256) -> InferenceResult:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        raw = generate_chat_completion(
            self.model,
            self.tokenizer,
            messages,
            max_new_tokens=max_new_tokens,
        )
        return parse_planner_response(raw)


def build_parser() -> argparse.ArgumentParser:
    default_model = load_training_config().model_name
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--load-in-4bit", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    planner = PlannerInference(
        args.model, args.adapter, load_in_4bit=args.load_in_4bit
    )
    result = planner.generate(args.question, max_new_tokens=args.max_new_tokens)
    print("Raw response:")
    print(result.raw_response)
    print("Parsed response:")
    print(json.dumps(result.parsed_response, ensure_ascii=False, indent=2))
    print(f"Schema valid: {result.schema_valid}")
    if result.error:
        print(f"Error: {result.error}")


if __name__ == "__main__":
    main()
