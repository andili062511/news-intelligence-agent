# News Intelligence Agent

A multi-source news intelligence agent based on RAG, tool orchestration and large language models.

Project Status: Under Development

Development Roadmap:

- [x] FastAPI backend
- [x] News ingestion pipeline
- [x] BM25 retrieval
- [x] FAISS vector retrieval
- [x] Hybrid retrieval
- [x] Cross-encoder reranking
- [x] LangGraph Agent
- [x] Evidence verification
- [x] Grounded Qwen generation
- [ ] QLoRA fine-tuning
- [ ] Evaluation
- [ ] Docker deployment

Architecture:

User Query

Planner

BM25 + FAISS

RRF

Cross-Encoder

Evidence Sufficiency

Citation Metadata Validation

Grounding Pack

Qwen Instruct (structured claims JSON)

Claim Citation Validation

Final Answer

## News Ingestion

The ingestion pipeline uses the following flow:

GDELT/RSS -> Full-text extraction -> Cleaning -> Deduplication -> Chunking -> JSONL

Run it with:

```bash
python -m ingestion.pipeline --query "artificial intelligence" --max-records 30 --timespan 1week --language English --provider auto
```

The pipeline writes UTF-8 JSONL files to `data/articles.jsonl` and
`data/chunks.jsonl`. It uses a small, sequential fetch limit for local
development and falls back to the GDELT summary or title when full-text
extraction is unavailable. English is selected by default; pass
`--language all` to retain articles in every language.
Use `--provider gdelt` or `--provider rss` to force one source. The default
`--provider auto` tries GDELT first and falls back to RSS when GDELT fails or
returns no articles.

The ingestion pipeline preserves the last valid local dataset when an upstream news provider is unavailable or rate-limited.

## BM25 Retrieval

BM25 provides keyword-based lexical retrieval over news chunks.

Run it with:

```bash
python -m retrieval.cli --query "artificial intelligence" --top-k 3
```

## FAISS Semantic Retrieval

Dense semantic embeddings are generated using Sentence Transformers and indexed
with FAISS. BM25 performs lexical / keyword retrieval, while FAISS performs
semantic / dense vector retrieval.

Run it with:

```bash
python -m retrieval.vector_cli --query "AI safety and advanced models" --top-k 3
```

## Hybrid Retrieval

The system combines BM25 lexical retrieval and FAISS dense semantic retrieval
using Reciprocal Rank Fusion (RRF). BM25 and cosine similarity operate on
different score scales, so rank-based fusion is used instead of directly
adding their scores. For each candidate, the fusion score is
`sum(1 / (rrf_k + rank))` across the retrieval routes that found it; the
default `rrf_k` is 60.

Run it with:

```bash
python -m retrieval.hybrid_cli --query "AI safety and risks of advanced models" --top-k 3
```

Use `--candidate-k` to control how many candidates each retriever contributes
before fusion. Results include the final `rank`, `rrf_score`, chunk metadata,
and the original BM25/FAISS ranks in `retrieval_sources`.

To improve source-article diversity by returning at most one chunk per
article:

```bash
python -m retrieval.hybrid_cli --query "AI safety and risks of advanced models" --top-k 3 --max-chunks-per-article 1
```

## Cross-Encoder Reranking

The final retrieval pipeline is:

BM25 + FAISS -> RRF -> Candidate Retrieval -> Cross-Encoder -> Final Evidence Ranking

BM25 and FAISS provide high-recall candidate retrieval. The cross-encoder then
scores each query-document pair more precisely to improve the ordering of the
final evidence. Article-level diversity is applied after reranking, so relevant
chunks are not discarded before the cross-encoder can evaluate them.

Run it with:

```bash
python -m retrieval.rerank_cli --query "machines becoming dangerously capable" --top-k 3 --candidate-k 15 --max-chunks-per-article 1
```

## LangGraph Planner-Executor Agent

LangGraph manages explicit agent state, conditional routing, bounded retries,
and deterministic evidence sufficiency. The first version uses a rule-based
planner and query rewriter, then generates an answer only from validated
evidence.

The agent workflow is:

Question -> Planner -> Search Tool -> Hybrid Retrieval + Reranker -> Evidence Check

When evidence is insufficient, the graph rewrites the query and retries search
up to the configured maximum. Sufficient evidence then passes through citation
preparation and validation. The graph finishes with `completed`,
`citation_failed`, or `insufficient_evidence`.

Run it with:

```bash
python -m agent.cli --question "What are the risks of rapidly advancing AI?" --model Qwen/Qwen2.5-1.5B-Instruct
```

## Evidence Verification & Citation Guardrails

The pre-generation evidence flow is:

Retrieval -> Reranking -> Evidence Sufficiency -> Deduplication -> Citation
Metadata Validation -> Grounding Pack -> Qwen Instruct JSON -> Claim Citation
Validation -> Deterministic Renderer -> Final Answer

Each retained evidence item receives a stable ranking-order citation ID (`E1`,
`E2`, ...). Required source metadata is validated, exact citation duplicates
are removed, and the grounding decision requires enough evidence from multiple
articles. No absolute reranker score threshold is used.

## Grounded Answer Generation

The language model receives only validated evidence and returns a JSON `claims`
array. Every claim carries its own evidence IDs, such as `E1` and `E2`. The
program validates those IDs and renders the final inline citations; it never
guesses a citation for a claim. Malformed JSON, empty claims, missing citations,
and unknown IDs receive at most one repair attempt. The final answer is exposed
only after structured validation passes.

`Qwen/Qwen2.5-1.5B-Instruct` can be used for local development. The architecture
supports `Qwen/Qwen2.5-7B-Instruct`, but successful 7B execution has not been
claimed or verified on appropriate hardware.
