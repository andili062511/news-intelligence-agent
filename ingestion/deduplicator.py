"""Rule-based article deduplication."""

from collections.abc import Mapping
from typing import Any

from .cleaner import normalize_title


def deduplicate_articles(
    articles: list[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Remove exact URL duplicates, then exact normalized-title duplicates."""

    before = len(articles)
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    unique: list[Mapping[str, Any]] = []

    for article in articles:
        url = str(article.get("url") or "").strip()
        title = normalize_title(str(article.get("title") or ""))
        if (url and url in seen_urls) or (title and title in seen_titles):
            continue

        if url:
            seen_urls.add(url)
        if title:
            seen_titles.add(title)
        unique.append(article)

    removed = before - len(unique)
    print(f"Before deduplication: {before}")
    print(f"After deduplication: {len(unique)}")
    print(f"Removed duplicates: {removed}")
    return unique

