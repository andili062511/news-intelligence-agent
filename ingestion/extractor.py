"""Full-text extraction for article URLs."""

import logging

import requests
import trafilatura


LOGGER = logging.getLogger(__name__)


def extract_article_text(url: str) -> str:
    """Download a public article page and extract its main text.

    This function does not attempt to bypass authentication, paywalls,
    robots restrictions, or other access controls.
    """

    if not url:
        LOGGER.warning("Cannot extract article text without a URL")
        return ""

    try:
        response = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": "NewsIntelligenceAgent/0.1"},
        )
        response.raise_for_status()
        text = trafilatura.extract(
            response.text,
            url=url,
            include_comments=False,
            include_tables=False,
        )
        if not text:
            LOGGER.warning("Trafilatura could not extract article text: %s", url)
            return ""
        return text
    except (requests.RequestException, ValueError, TypeError) as exc:
        LOGGER.warning("Unable to extract article text from %s: %s", url, exc)
        return ""
    except Exception as exc:  # Trafilatura can raise parser-specific exceptions.
        LOGGER.warning("Article extraction failed for %s: %s", url, exc)
        return ""

