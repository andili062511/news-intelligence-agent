"""Offline tests for QLoRA configuration and dataset wiring."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from finetune.training_utils import (
    build_quantization_config,
    build_sft_config,
    get_compute_dtype,
    load_planner_datasets,
    load_training_config,
)


EXPECTED_TARGETS = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}


def test_qlora_config_values_and_target_modules() -> None:
    config = load_training_config()

    assert config.model_name == "Qwen/Qwen2.5-1.5B-Instruct"
    assert config.quantization == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
    }
    assert config.lora["r"] == 16
    assert config.lora["lora_alpha"] == 32
    assert config.lora["lora_dropout"] == 0.05
    assert config.lora["bias"] == "none"
    assert config.lora["task_type"] == "CAUSAL_LM"
    assert set(config.lora["target_modules"]) == EXPECTED_TARGETS
    assert config.training["max_length"] == 512
    assert config.training["eval_strategy"] == "steps"
    assert config.training["eos_token"] == "<|im_end|>"


class _FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def is_bf16_supported() -> bool:
        return True


class _FakeTorch:
    cuda = _FakeCuda()
    bfloat16 = "bf16"
    float16 = "fp16"


def test_compute_dtype_prefers_bf16_when_supported() -> None:
    assert get_compute_dtype(_FakeTorch) == "bf16"


class _EmulatedBf16Cuda:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def is_bf16_supported(including_emulation: bool = True) -> bool:
        return including_emulation


class _TuringLikeTorch:
    cuda = _EmulatedBf16Cuda()
    bfloat16 = "bf16"
    float16 = "fp16"


def test_compute_dtype_rejects_emulated_bf16() -> None:
    assert get_compute_dtype(_TuringLikeTorch) == "fp16"


def test_installed_sft_config_accepts_all_training_arguments() -> None:
    config = load_training_config()
    sft_config = build_sft_config(config.training)

    assert sft_config.warmup_steps == 0.05
    assert sft_config.eval_strategy.value == "steps"
    assert sft_config.max_length == 512


def test_nf4_bitsandbytes_config_builder() -> None:
    config = load_training_config()
    quantization = build_quantization_config(config.quantization)

    assert quantization.load_in_4bit is True
    assert quantization.bnb_4bit_quant_type == "nf4"
    assert quantization.bnb_4bit_use_double_quant is True


def test_dataset_loading_uses_hugging_face_dataset_without_network(
    tmp_path: Path, monkeypatch
) -> None:
    output = {
        "intent": "news_search",
        "search_queries": ["OpenAI news"],
        "time_range": None,
        "tools": ["news_search"],
    }
    sample = {
        "messages": [
            {"role": "system", "content": "Plan news retrieval."},
            {"role": "user", "content": "What is new with OpenAI?"},
            {"role": "assistant", "content": json.dumps(output)},
        ]
    }
    train_file = tmp_path / "train.jsonl"
    val_file = tmp_path / "val.jsonl"
    train_file.write_text(json.dumps(sample) + "\n", encoding="utf-8")
    val_file.write_text(json.dumps(sample) + "\n", encoding="utf-8")

    class FakeSplit(list):
        column_names = ["messages"]

    calls: dict[str, object] = {}

    def fake_load_dataset(format_name: str, *, data_files: dict[str, str]):
        calls.update(format_name=format_name, data_files=data_files)
        return {"train": FakeSplit([sample]), "validation": FakeSplit([sample])}

    monkeypatch.setitem(
        sys.modules, "datasets", types.SimpleNamespace(load_dataset=fake_load_dataset)
    )
    loaded = load_planner_datasets(train_file, val_file)

    assert calls["format_name"] == "json"
    assert len(loaded["train"]) == 1
    assert len(loaded["validation"]) == 1
