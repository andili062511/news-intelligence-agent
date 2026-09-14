"""Command-line pipeline for news ingestion."""

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .chunker import create_chunks
from .cleaner import clean_text
from .deduplicator import deduplicate_articles
from .extractor import extract_article_text
from .providers import fetch_news


LOGGER = logging.getLogger(__name__)
DEFAULT_OUTPUT_DIR = Path("data")
NO_ARTICLES_WARNING = (
    "WARNING: No articles fetched. Existing dataset has been preserved."
)


class IngestionError(RuntimeError):
    """Raised when ingestion cannot safely publish a new dataset."""


ENGLISH_LANGUAGE_ALIASES = {
    "en",
    "eng",
    "english",
    "en-gb",
    "en-us",
}


def _normalize_language(value: str) -> str:
    """Normalize language names and common English language codes."""

    normalized = value.strip().lower().replace("_", "-")
    if normalized in ENGLISH_LANGUAGE_ALIASES or normalized.startswith("en-"):
        return "english"
    return normalized


def _language_matches(article_language: str, requested_language: str) -> bool:
    """Return whether an article matches the requested ingestion language."""

    requested = _normalize_language(requested_language)
    return requested == "all" or _normalize_language(article_language) == requested


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
        try:
            return parsedate_to_datetime(raw).isoformat()
        except (TypeError, ValueError, OverflowError):
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
        output_file.flush()
        os.fsync(output_file.fileno())


def _validate_staged_jsonl(path: Path, expected_records: int) -> None:
    """Verify that a staged JSONL file is non-empty and fully readable."""

    if expected_records <= 0 or not path.is_file() or path.stat().st_size == 0:
        raise IngestionError(f"Refusing to publish an empty JSONL file: {path}")

    with path.open("r", encoding="utf-8") as input_file:
        records = [json.loads(line) for line in input_file if line.strip()]
    if len(records) != expected_records or not all(
        isinstance(record, dict) for record in records
    ):
        raise IngestionError(f"Staged JSONL validation failed: {path}")


def _cleanup_staging_files(output_dir: Path) -> None:
    """Remove only the pipeline's known staging and rollback files."""

    for filename in (
        "articles.jsonl.tmp",
        "chunks.jsonl.tmp",
        "articles.jsonl.backup",
        "chunks.jsonl.backup",
    ):
        (output_dir / filename).unlink(missing_ok=True)


def _publish_dataset(
    output_dir: Path,
    articles: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> None:
    """Stage both JSONL files before atomically replacing their destinations."""

    articles_path = output_dir / "articles.jsonl"
    chunks_path = output_dir / "chunks.jsonl"
    articles_tmp = output_dir / "articles.jsonl.tmp"
    chunks_tmp = output_dir / "chunks.jsonl.tmp"
    articles_backup = output_dir / "articles.jsonl.backup"
    chunks_backup = output_dir / "chunks.jsonl.backup"
    backup_paths = (articles_backup, chunks_backup)
    destinations = (articles_path, chunks_path)
    destination_existed = tuple(path.exists() for path in destinations)
    publish_started = False

    try:
        _cleanup_staging_files(output_dir)
        if not articles or not chunks:
            raise IngestionError("Refusing to publish an empty dataset")

        _write_jsonl(articles_tmp, articles)
        _write_jsonl(chunks_tmp, chunks)
        _validate_staged_jsonl(articles_tmp, len(articles))
        _validate_staged_jsonl(chunks_tmp, len(chunks))

        for destination, backup, existed in zip(
            destinations, backup_paths, destination_existed
        ):
            if existed:
                shutil.copy2(destination, backup)

        publish_started = True
        os.replace(articles_tmp, articles_path)
        os.replace(chunks_tmp, chunks_path)
    except Exception:
        if publish_started:
            for destination, backup, existed in zip(
                destinations, backup_paths, destination_existed
            ):
                if existed and backup.exists():
                    os.replace(backup, destination)
                elif not existed:
                    destination.unlink(missing_ok=True)
        raise
    finally:
        _cleanup_staging_files(output_dir)


def run_pipeline(
    query: str,
    max_records: int = 20,
    timespan: str = "1week",
    language: str = "English",
    provider: str = "auto",
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run ingestion and write article/chunk JSONL files."""

    _cleanup_staging_files(output_dir)
    raw_articles, _provider_used = fetch_news(
        query=query,
        max_records=max_records,
        timespan=timespan,
        provider=provider,
    )
    if not raw_articles:
        print(NO_ARTICLES_WARNING)
        raise IngestionError("Upstream news fetch returned no articles")

    fetched_at = datetime.now(timezone.utc).isoformat()
    standardized = [
        article
        for raw in raw_articles
        if isinstance(raw, dict)
        for article in [_standardize_article(raw, fetched_at)]
        if article is not None
    ]
    language_matched = [
        article
        for article in standardized
        if _language_matches(article["language"], language)
    ]
    unique_articles = [dict(article) for article in deduplicate_articles(language_matched)]
    chunks = [chunk for article in unique_articles for chunk in create_chunks(article)]

    if not unique_articles or not chunks:
        print(
            "WARNING: Ingestion produced no usable dataset. "
            "Existing dataset has been preserved."
        )
        raise IngestionError("Ingestion produced no usable articles or chunks")

    _publish_dataset(output_dir, unique_articles, chunks)

    print(f"Fetched: {len(raw_articles)}")
    print(f"Language matched: {len(language_matched)}")
    print(f"Valid articles: {len(standardized)}")
    print(f"Duplicates removed: {len(language_matched) - len(unique_articles)}")
    print(f"Articles saved: {len(unique_articles)}")
    print(f"Chunks created: {len(chunks)}")
    return unique_articles, chunks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest news articles")
    parser.add_argument("--query", required=True, help="News search query")
    parser.add_argument("--max-records", type=int, default=20)
    parser.add_argument("--timespan", default="1week")
    parser.add_argument(
        "--provider",
        choices=["auto", "gdelt", "rss"],
        default="auto",
    )
    parser.add_argument(
        "--language",
        default="English",
        help='Article language to retain, or "all" to disable language filtering',
    )
    return parser


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args()
    try:
        run_pipeline(
            args.query,
            max_records=args.max_records,
            timespan=args.timespan,
            language=args.language,
            provider=args.provider,
        )
    except IngestionError as exc:
        LOGGER.error("Ingestion failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

