# News Intelligence Agent

A multi-source news intelligence agent combining hybrid RAG, tool orchestration, evidence validation and parameter-efficient planner fine-tuning.

## Architecture

```text
User Question
    -> Planner
    -> BM25 + FAISS
    -> Reciprocal Rank Fusion (RRF)
    -> Cross-Encoder
    -> Evidence Sufficiency
    -> Citation Validation
    -> Qwen Grounded Generation
    -> Final Answer
```

When evidence is insufficient, the LangGraph workflow rewrites the query and retries within a fixed limit. A final answer is released only after its evidence metadata and claim citations pass validation.

## Key Features

- Multi-source ingestion with retry and GDELT-to-RSS fallback
- Cleaning, deduplication and chunking
- Hybrid lexical and semantic retrieval with BM25 and FAISS
- Reciprocal Rank Fusion and Cross-Encoder reranking
- LangGraph stateful agent orchestration
- Evidence sufficiency and citation guardrails
- Grounded Qwen answer generation
- QLoRA / PEFT planner fine-tuning
- FastAPI API with interactive documentation
- Docker deployment

## Fine-tuning

The planner was fine-tuned from `Qwen/Qwen2.5-1.5B-Instruct` using 4-bit QLoRA and PEFT. Fine-tuning targets structured planner behavior—intent recognition, time-range mapping, query construction and tool selection—not dynamic news knowledge.

The project-generated dataset contains 900 training samples and 100 validation samples. The local smoke and mini-training experiment ran on an RTX 2060 6 GB GPU. The resulting adapter has 18,464,768 trainable parameters out of 907,081,216 total parameters (2.0356%). See [reports/FINAL_EVALUATION.md](reports/FINAL_EVALUATION.md) for the recorded experiment results.

## Evaluation

Hard evaluation uses 100 project-generated samples.

| Metric | Base | QLoRA |
|---|---:|---:|
| Intent Accuracy | 0% | 87% |
| Tool Exact Match | 0% | 80% |
| Time Range Accuracy | 26% | 72% |
| JSON Valid Rate | 40% | 100% |
| Schema Valid Rate | 0% | 100% |
| Tool F1 | - | 94% |

QLoRA produced the largest gains in structured output validity, intent recognition and tool selection. Exact time-range mapping and query wording remain the main weaknesses.

## Quick Start

Create and activate a virtual environment:

```bash
python -m venv .venv
```

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Ingest news from GDELT with automatic RSS fallback:

```bash
python -m ingestion.pipeline --query "artificial intelligence" --max-records 30 --timespan 1week --language English --provider auto
```

Run the retrieval stages directly:

```bash
python -m retrieval.cli --query "artificial intelligence" --top-k 3
python -m retrieval.vector_cli --query "AI safety and advanced models" --top-k 3
python -m retrieval.hybrid_cli --query "AI safety risks" --top-k 3
python -m retrieval.rerank_cli --query "AI safety risks" --top-k 3 --candidate-k 15
```

Run the LangGraph agent CLI:

```bash
python -m agent.cli --question "What are the risks of rapidly advancing AI?" --model Qwen/Qwen2.5-1.5B-Instruct
```

Start FastAPI:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/docs` to test `POST /query`, or check `GET /health`.

Example request:

```bash
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"question":"What are the risks of rapidly advancing AI?"}'
```

Build and run with Docker:

```bash
docker build -t news-intelligence-agent .
docker run -p 8000:8000 news-intelligence-agent
```

The default image is intended to start the API with a CPU-compatible architecture. It does not claim or perform GPU QLoRA training inside the image. The first real query loads the configured retrieval and generation models; set `NIA_MODEL` only when intentionally selecting a compatible answer model.

## API

`POST /query` reuses the existing LangGraph agent, hybrid retrieval pipeline and Qwen generator. Its response includes the question, terminal status, validated answer, citations, evidence count, intent and final search queries. `insufficient_evidence` returns no answer, and `citation_failed` never exposes an unvalidated model answer.

## Project Structure

```text
app/                 FastAPI application
agent/               LangGraph planner-executor and validation guardrails
ingestion/           GDELT/RSS ingestion, cleaning and chunking
retrieval/           BM25, FAISS, RRF and Cross-Encoder retrieval
llm/                 Grounded Qwen generation backend
finetune/            Planner dataset, QLoRA utilities and evaluation
finetune/data/       Retained instruction and evaluation JSONL datasets
reports/             Final evaluation report and hard-eval artifacts
tests/               Offline unit and integration tests
Dockerfile           API container definition
requirements.txt     Python dependencies
```

Generated articles, chunks, model weights, adapters, checkpoints and training outputs are intentionally excluded from Git and the Docker build context.

## Limitations

- The evaluation dataset is project-generated.
- Dynamic news knowledge is retrieved through RAG rather than memorized through fine-tuning.
- The local fine-tuning experiment used `Qwen/Qwen2.5-1.5B-Instruct`.
- Generated news quality still depends on upstream source availability and evidence quality.
- CPU inference is supported architecturally but may be slow and memory-intensive.

## Roadmap

- [x] FastAPI backend and `/query` endpoint
- [x] Multi-source news ingestion with retry and fallback
- [x] Cleaning, deduplication and chunking
- [x] BM25 and FAISS retrieval
- [x] RRF hybrid retrieval
- [x] Cross-Encoder reranking
- [x] LangGraph planner-executor
- [x] Evidence sufficiency and query retry
- [x] Citation validation and grounded generation
- [x] QLoRA / PEFT planner fine-tuning
- [x] Standard and hard evaluation
- [x] Docker packaging

Future work:

- Larger-scale planner fine-tuning
- More multilingual retrieval
- NLI-based claim verification
