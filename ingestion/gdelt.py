"""Client for the GDELT DOC 2.0 API."""

import logging
from typing import Any

import requests


LOGGER = logging.getLogger(__name__)
GDELT_DOC_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


def fetch_articles(
    query: str,
    max_records: int = 50,
    timespan: str = "1week",
) -> list[dict[str, Any]]:
    """Fetch article metadata from GDELT.

    Network and malformed-response errors are logged and converted into an
    empty list so callers can continue gracefully.
    """

    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": max_records,
        "timespan": timespan,
        "sort": "datedesc",
    }

    try:
        response = requests.get(GDELT_DOC_API_URL, params=params, timeout=15)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("GDELT response is not a JSON object")
        articles = payload.get("articles", [])
        if not isinstance(articles, list):
            raise ValueError("GDELT response field 'articles' is not a list")
    except Exception as exc:
        LOGGER.warning("Unable to fetch articles from GDELT: %s", exc)
        print("Fetched articles: 0")
        return []

    print(f"Fetched articles: {len(articles)}")
    return articles
