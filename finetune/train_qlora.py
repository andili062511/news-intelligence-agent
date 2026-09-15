"""Fine-tune the News Intelligence Agent planner with QLoRA and TRL."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from finetune.training_utils import (
    DEFAULT_CONFIG_PATH,
    build_lora_config,
    build_quantization_config,
    build_sft_config,
    count_trainable_parameters,
    get_compute_dtype,
    load_planner_datasets,
    load_training_config,
    print_trainable_parameters,
)


def build_parser() -> argparse.ArgumentParser:
    defaults = load_training_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model", default=defaults.model_name)
    parser.add_argument(
        "--train-file", type=Path, default=Path("finetune/data/planner_train.jsonl")
    )
    parser.add_argument(
        "--val-file", type=Path, default=Path("finetune/data/planner_val.jsonl")
    )
    parser.add_argument("--output-dir", type=Path, default=Path(defaults.training["output_dir"]))
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override epoch-based training (useful for a small-model smoke test).",
    )
    return parser


def _device_map(torch_module: Any) -> dict[str, int] | None:
    """Pin a quantized model to this process's CUDA device when CUDA is available."""

    if not torch_module.cuda.is_available():
        return None
    import os

    return {"": int(os.environ.get("LOCAL_RANK", "0"))}


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    """Construct and train the model. Called explicitly by the CLI, never on import."""

    import torch
    import bitsandbytes
    import peft
    import transformers
    import trl
    from peft import get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    config = load_training_config(args.config)
    dtype = get_compute_dtype(torch)
    training_values = dict(config.training)
    training_values["output_dir"] = str(args.output_dir)
    if args.max_steps is not None:
        if args.max_steps <= 0:
            raise ValueError("--max-steps must be greater than zero")
        training_values["max_steps"] = args.max_steps

    # Validate against the installed TRL API before loading a tokenizer or model.
    sft_config = build_sft_config(
        training_values, sft_config_class=SFTConfig
    )

    print(f"torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    print(f"GPU name: {gpu_name}")
    print(f"TRL version: {trl.__version__}")
    print(f"Transformers version: {transformers.__version__}")
    print(f"PEFT version: {peft.__version__}")
    print(f"bitsandbytes version: {bitsandbytes.__version__}")
    print(f"compute dtype: {dtype}")

    datasets = load_planner_datasets(args.train_file, args.val_file)
    quantization_config = build_quantization_config(
        config.quantization, torch_module=torch
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization_config,
        dtype=dtype,
        device_map=_device_map(torch),
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=bool(config.training["gradient_checkpointing"]),
    )
    model = get_peft_model(model, build_lora_config(config.lora))

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=tokenizer,
    )
    print_trainable_parameters(trainer.model)
    trainable, total, trainable_percentage = count_trainable_parameters(trainer.model)
    train_result = trainer.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    metrics = dict(getattr(train_result, "metrics", {}))
    metadata = {
        "base_model": args.model,
        "adapter_type": "QLoRA",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_file": str(args.train_file),
        "validation_file": str(args.val_file),
        "train_samples": len(datasets["train"]),
        "validation_samples": len(datasets["validation"]),
        "compute_dtype": str(dtype),
        "parameters": {
            "trainable": trainable,
            "total": total,
            "trainable_percentage": trainable_percentage,
        },
        "quantization": config.quantization,
        "lora": config.lora,
        "training": training_values,
        "train_metrics": metrics,
    }
    with (args.output_dir / "training_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, default=str)
    return metadata


def main() -> None:
    args = build_parser().parse_args()
    metadata = run_training(args)
    print(f"Adapter and tokenizer saved to: {args.output_dir}")
    print(f"Training samples: {metadata['train_samples']}")
    print(f"Validation samples: {metadata['validation_samples']}")


if __name__ == "__main__":
    main()
