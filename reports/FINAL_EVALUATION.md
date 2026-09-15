# Planner Fine-tuning Evaluation

## Model

Base:
`Qwen/Qwen2.5-1.5B-Instruct`

Fine-tuned:
`Qwen/Qwen2.5-1.5B-Instruct` + QLoRA/PEFT

## Fine-tuning

- Training samples: 900
- Validation samples: 100
- Local smoke / mini training: RTX 2060 6GB
- Trainable parameters: 18,464,768 / 907,081,216 (2.0356%)

50-step training:

- `train_loss`: 0.7025
- `eval_loss`: 0.1086

Training loss is recorded for reproducibility but is not treated as the primary project metric. The primary metrics evaluate whether the planner produces valid, structurally correct retrieval plans.

## Standard Validation

| Metric | Base | QLoRA |
|---|---:|---:|
| Intent Accuracy | 0.00 | 0.98 |
| Tool Exact Match | 0.00 | 0.94 |
| Time Range Accuracy | 0.35 | 0.95 |
| JSON Valid Rate | 0.52 | 1.00 |
| Schema Valid Rate | 0.00 | 1.00 |
| Full Exact Match | 0.00 | 0.86 |

## Hard Evaluation

The hard evaluation contains 100 samples.

| Metric | Base | QLoRA |
|---|---:|---:|
| Intent Accuracy | 0.00 | 0.87 |
| Tool Exact Match | 0.00 | 0.80 |
| Time Range Accuracy | 0.26 | 0.72 |
| JSON Valid Rate | 0.40 | 1.00 |
| Schema Valid Rate | 0.00 | 1.00 |
| Search Query Non-empty Rate | — | 1.00 |
| Structural Exact Match | — | 0.66 |
| Tool Precision | — | 0.99 |
| Tool Recall | — | 0.90 |
| Tool F1 | — | 0.94 |
| Entity Coverage | — | 0.95 |
| Query Token Recall | — | 0.58 |

## Analysis

QLoRA strongly improved structured planner behavior. The strongest gains were JSON/schema validity, intent recognition and tool selection. Remaining weaknesses were precise time-range mapping and exact query wording.

Full Exact Match is intentionally retained for continuity, but it is overly strict for semantically equivalent search queries. Structural and component-level metrics therefore provide a more informative view of planner quality while preserving the strict metric as a reference.
