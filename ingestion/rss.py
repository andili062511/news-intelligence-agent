"""RSS news provider used when GDELT is unavailable."""

import logging
import re
import xml.etree.ElementTree as ET
from typing import Any

import requests


LOGGER = logging.getLogger(__name__)
GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"


def _google_when(timespan: str) -> str:
    """Translate common GDELT timespans into a Google News ``when`` value."""

    match = re.fullmatch(
        r"\s*(\d+)\s*(minute|hour|day|week|month|year)s?\s*",
        timespan,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""

    amount = int(match.group(1))
    unit = match.group(2).lower()
    if unit == "minute":
        return f"{amount}m"
    if unit == "hour":
        return f"{amount}h"
    days = amount * {"day": 1, "week": 7, "month": 30, "year": 365}[unit]
    return f"{days}d"


def _child_text(element: ET.Element, name: str) -> str:
    child = element.find(name)
    return (child.text or "").strip() if child is not None else ""


def fetch_articles(
    query: str,
    max_records: int = 50,
    timespan: str = "1week",
) -> list[dict[str, Any]]:
    """Fetch Google News RSS items as provider-neutral article dictionaries."""

    if max_records <= 0:
        return []

    when = _google_when(timespan)
    rss_query = f"{query} when:{when}" if when else query
    try:
        response = requests.get(
            GOOGLE_NEWS_RSS_URL,
            params={"q": rss_query, "hl": "en-US", "gl": "US", "ceid": "US:en"},
            timeout=15,
            headers={"User-Agent": "NewsIntelligenceAgent/0.1"},
        )
        response.raise_for_status()
        root = ET.fromstring(response.content)
    except Exception as exc:
        LOGGER.warning("Unable to fetch articles from RSS: %s", exc)
        return []

    channel = root.find("channel")
    channel_language = _child_text(channel, "language") if channel is not None else ""
    articles: list[dict[str, Any]] = []
    for item in root.findall("./channel/item")[:max_records]:
        source_element = item.find("source")
        source = (
            (source_element.text or "").strip()
            if source_element is not None
            else ""
        )
        articles.append(
            {
                "title": _child_text(item, "title"),
                "url": _child_text(item, "link"),
                "source": source,
                "published_at": _child_text(item, "pubDate"),
                "language": channel_language or "English",
                "summary": _child_text(item, "description"),
            }
        )
    return articles
