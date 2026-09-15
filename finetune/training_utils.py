"""Shared, import-safe helpers for planner QLoRA training.

Heavy ML dependencies are imported inside the functions that use them. Merely
importing this module never downloads or constructs a model.
"""

from __future__ import annotations

import json
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from finetune.schemas import validate_sample


DEFAULT_CONFIG_PATH = Path(__file__).parent / "configs" / "qlora_qwen2_5_7b.json"


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    model_name: str
    quantization: dict[str, Any]
    lora: dict[str, Any]
    training: dict[str, Any]


def load_training_config(path: Path | str = DEFAULT_CONFIG_PATH) -> TrainingConfig:
    """Load and minimally validate the centralized QLoRA configuration."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    required = {"model_name", "quantization", "lora", "training"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"training config must contain exactly {sorted(required)}")
    for section in ("quantization", "lora", "training"):
        if not isinstance(value[section], dict):
            raise ValueError(f"{section} must be a JSON object")
    if not isinstance(value["model_name"], str) or not value["model_name"].strip():
        raise ValueError("model_name must be a non-empty string")
    return TrainingConfig(
        model_name=value["model_name"],
        quantization=dict(value["quantization"]),
        lora=dict(value["lora"]),
        training=dict(value["training"]),
    )


def get_compute_dtype(torch_module: Any | None = None) -> Any:
    """Return native bfloat16 on capable CUDA hardware, otherwise float16."""

    if torch_module is None:
        import torch as torch_module

    cuda = getattr(torch_module, "cuda", None)
    bf16_supported = False
    if (
        cuda is not None
        and cuda.is_available()
        and hasattr(cuda, "is_bf16_supported")
    ):
        check_bf16 = cuda.is_bf16_supported
        # Recent PyTorch versions include BF16 emulation in the default result.
        # Quantization needs the GPU's native dtype (RTX 2060/Turing is FP16).
        if "including_emulation" in inspect.signature(check_bf16).parameters:
            bf16_supported = bool(check_bf16(including_emulation=False))
        else:
            bf16_supported = bool(check_bf16())
    return torch_module.bfloat16 if bf16_supported else torch_module.float16


def build_sft_config(
    values: Mapping[str, Any], *, sft_config_class: Any | None = None
) -> Any:
    """Validate values against the installed TRL SFTConfig API and construct it."""

    if sft_config_class is None:
        from trl import SFTConfig as sft_config_class

    options = dict(values)
    parameters = inspect.signature(sft_config_class).parameters

    # TRL 1.13 removed warmup_ratio. Its float-valued warmup_steps accepts a
    # fraction below 1.0, preserving the ratio semantics in Transformers.
    if "warmup_ratio" in options and "warmup_ratio" not in parameters:
        if "warmup_steps" not in parameters:
            raise TypeError(
                "installed TRL SFTConfig supports neither warmup_ratio nor "
                "the equivalent warmup_steps"
            )
        options["warmup_steps"] = options.pop("warmup_ratio")

    unsupported = sorted(set(options) - set(parameters))
    if unsupported:
        raise TypeError(
            "unsupported arguments for installed TRL SFTConfig: "
            + ", ".join(unsupported)
        )
    return sft_config_class(**options)


def build_quantization_config(
    values: Mapping[str, Any], *, torch_module: Any | None = None
) -> Any:
    """Build the Transformers NF4 configuration with a detected compute dtype."""

    from transformers import BitsAndBytesConfig

    options = dict(values)
    options["bnb_4bit_compute_dtype"] = get_compute_dtype(torch_module)
    return BitsAndBytesConfig(**options)


def build_lora_config(values: Mapping[str, Any]) -> Any:
    """Build the PEFT LoRA configuration from the central config file."""

    from peft import LoraConfig

    return LoraConfig(**dict(values))


def load_planner_datasets(train_file: Path | str, val_file: Path | str) -> Any:
    """Load validated conversational JSONL files as a Hugging Face DatasetDict."""

    train_path = Path(train_file)
    val_path = Path(val_file)
    for path in (train_path, val_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    from datasets import load_dataset

    datasets = load_dataset(
        "json",
        data_files={"train": str(train_path), "validation": str(val_path)},
    )
    for split_name in ("train", "validation"):
        split = datasets[split_name]
        if "messages" not in split.column_names:
            raise ValueError(f"{split_name} dataset must contain a messages column")
        for row_number, sample in enumerate(split, start=1):
            try:
                validate_sample(sample)
            except ValueError as exc:
                raise ValueError(f"{split_name} row {row_number}: {exc}") from exc
    return datasets


def count_trainable_parameters(model: Any) -> tuple[int, int, float]:
    """Return trainable count, total count, and trainable percentage."""

    trainable = 0
    total = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    percentage = 100.0 * trainable / total if total else 0.0
    return trainable, total, percentage


def print_trainable_parameters(model: Any) -> None:
    trainable, total, percentage = count_trainable_parameters(model)
    print(
        f"Trainable parameters: {trainable:,} / {total:,} "
        f"({percentage:.4f}%)"
    )
