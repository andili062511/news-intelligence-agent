"""Command-line pipeline for GDELT news ingestion."""

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .chunker import create_chunks
from .cleaner import clean_text
from .deduplicator import deduplicate_articles
from .extractor import extract_article_text
from .gdelt import fetch_articles


LOGGER = logging.getLogger(__name__)
DEFAULT_OUTPUT_DIR = Path("data")


def _to_iso8601(value: Any) -> str:
    """Convert common GDELT date values to an ISO 8601 string when possible."""

    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat()

    raw = str(value).strip()
    if not raw:
        return ""
    if raw.endswith("Z"):
        try:
            return datetime.fromisoformat(raw[:-1] + "+00:00").isoformat()
        except ValueError:
            pass

    for date_format in ("%Y%m%dT%H%M%SZ", "%Y%m%d%H%M%S", "%Y%m%dT%H%M%S"):
        try:
            parsed = datetime.strptime(raw, date_format)
            return parsed.replace(tzinfo=timezone.utc).isoformat()
        except ValueError:
            continue

    try:
        return datetime.fromisoformat(raw).isoformat()
    except ValueError:
        return raw


def _article_id(url: str, source: str, title: str, published_at: str) -> str:
    identity = url or f"{source}{title}{published_at}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def _standardize_article(raw: dict[str, Any], fetched_at: str) -> dict[str, Any] | None:
    """Map one GDELT record into the project Article schema."""

    url = str(raw.get("url") or "").strip()
    title = clean_text(str(raw.get("title") or ""))
    source = clean_text(
        str(raw.get("source") or raw.get("domain") or raw.get("sourcecountry") or "")
    )
    if not source and url:
        source = urlparse(url).netloc
    published_at = _to_iso8601(raw.get("published_at") or raw.get("seendate"))
    language = clean_text(str(raw.get("language") or ""))
    summary = clean_text(
        str(raw.get("summary") or raw.get("snippet") or raw.get("description") or "")
    )

    if not (url or title or summary):
        return None

    text = ""
    if url:
        text = clean_text(extract_article_text(url))
    if not text:
        text = summary or title

    return {
        "article_id": _article_id(url, source, title, published_at),
        "title": title,
        "source": source,
        "url": url,
        "published_at": published_at,
        "language": language,
        "summary": summary,
        "text": text,
        "fetched_at": fetched_at,
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_pipeline(
    query: str,
    max_records: int = 20,
    timespan: str = "1week",
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run ingestion and write article/chunk JSONL files."""

    raw_articles = fetch_articles(query, max_records=max_records, timespan=timespan)
    fetched_at = datetime.now(timezone.utc).isoformat()
    standardized = [
        article
        for raw in raw_articles
        if isinstance(raw, dict)
        for article in [_standardize_article(raw, fetched_at)]
        if article is not None
    ]
    unique_articles = [dict(article) for article in deduplicate_articles(standardized)]
    chunks = [chunk for article in unique_articles for chunk in create_chunks(article)]

    _write_jsonl(output_dir / "articles.jsonl", unique_articles)
    _write_jsonl(output_dir / "chunks.jsonl", chunks)

    print(f"Fetched: {len(raw_articles)}")
    print(f"Valid articles: {len(standardized)}")
    print(f"Duplicates removed: {len(standardized) - len(unique_articles)}")
    print(f"Articles saved: {len(unique_articles)}")
    print(f"Chunks created: {len(chunks)}")
    return unique_articles, chunks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest news articles from GDELT DOC 2.0")
    parser.add_argument("--query", required=True, help="GDELT search query")
    parser.add_argument("--max-records", type=int, default=20)
    parser.add_argument("--timespan", default="1week")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args()
    run_pipeline(args.query, max_records=args.max_records, timespan=args.timespan)


if __name__ == "__main__":
    main()

