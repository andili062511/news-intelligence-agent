# News Intelligence Agent

A multi-source news intelligence agent based on RAG, tool orchestration and large language models.

Project Status: Under Development

Development Roadmap:

- [x] FastAPI backend
- [x] News ingestion pipeline
- [x] BM25 retrieval
- [ ] FAISS vector retrieval
- [ ] Hybrid retrieval
- [ ] Cross-encoder reranking
- [ ] LangGraph Agent
- [ ] Evidence verification
- [ ] QLoRA fine-tuning
- [ ] Evaluation
- [ ] Docker deployment

Planned Architecture:

User Query

Planner

Hybrid Retrieval

BM25 + FAISS

Cross-Encoder Reranker

Evidence Checker

LLM

Citation Verification

Final Answer

## News Ingestion

The ingestion pipeline uses the following flow:

GDELT -> Full-text extraction -> Cleaning -> Deduplication -> Chunking -> JSONL

Run it with:

```bash
python -m ingestion.pipeline --query "artificial intelligence" --max-records 20 --timespan 1week
```

The pipeline writes UTF-8 JSONL files to `data/articles.jsonl` and
`data/chunks.jsonl`. It uses a small, sequential fetch limit for local
development and falls back to the GDELT summary or title when full-text
extraction is unavailable.

## BM25 Retrieval

BM25 provides keyword-based lexical retrieval over news chunks.

Run it with:

```bash
python -m retrieval.cli --query "artificial intelligence" --top-k 3
```
