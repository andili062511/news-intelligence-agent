"""Client for the GDELT DOC 2.0 API."""

import logging
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import requests


LOGGER = logging.getLogger(__name__)
GDELT_DOC_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
MAX_RETRIES = 3
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _retry_delay(response: requests.Response, retry_number: int) -> float:
    """Use Retry-After when valid, otherwise return exponential backoff."""

    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(retry_after)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError, OverflowError):
                pass
    return float(2**retry_number)


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
        for attempt in range(MAX_RETRIES + 1):
            response = requests.get(GDELT_DOC_API_URL, params=params, timeout=15)
            if (
                response.status_code in RETRYABLE_STATUS_CODES
                and attempt < MAX_RETRIES
            ):
                retry_number = attempt + 1
                delay = _retry_delay(response, retry_number)
                LOGGER.warning(
                    "GDELT request returned HTTP %d; retrying in %.1f seconds "
                    "(%d/%d)",
                    response.status_code,
                    delay,
                    retry_number,
                    MAX_RETRIES,
                )
                time.sleep(delay)
                continue

            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("GDELT response is not a JSON object")
            articles = payload.get("articles", [])
            if not isinstance(articles, list):
                raise ValueError("GDELT response field 'articles' is not a list")
            break
    except Exception as exc:
        LOGGER.warning("Unable to fetch articles from GDELT: %s", exc)
        print("Fetched articles: 0")
        return []

    print(f"Fetched articles: {len(articles)}")
    return articles
