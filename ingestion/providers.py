"""Unified selection and fallback for news metadata providers."""

import logging

from typing import Any, Literal

from .gdelt import fetch_articles as fetch_gdelt_articles
from .rss import fetch_articles as fetch_rss_articles


Provider = Literal["auto", "gdelt", "rss"]
LOGGER = logging.getLogger(__name__)


def fetch_news(
    query: str,
    max_records: int = 50,
    timespan: str = "1week",
    provider: Provider = "auto",
) -> tuple[list[dict[str, Any]], str]:
    """Fetch news using the selected provider and return records plus its name."""

    if provider == "gdelt":
        print("Primary provider: GDELT")
        articles = fetch_gdelt_articles(
            query, max_records=max_records, timespan=timespan
        )
        return articles, "GDELT"

    if provider == "rss":
        articles = fetch_rss_articles(query, max_records=max_records, timespan=timespan)
        print("Provider used: RSS")
        return articles, "RSS"

    if provider != "auto":
        raise ValueError(f"Unsupported provider: {provider}")

    print("Primary provider: GDELT")
    try:
        articles = fetch_gdelt_articles(
            query, max_records=max_records, timespan=timespan
        )
    except Exception as exc:
        LOGGER.warning("GDELT provider raised an exception: %s", exc)
        articles = []

    if articles:
        return articles, "GDELT"

    print("GDELT unavailable. Falling back to RSS.")
    articles = fetch_rss_articles(query, max_records=max_records, timespan=timespan)
    print("Provider used: RSS")
    return articles, "RSS"
