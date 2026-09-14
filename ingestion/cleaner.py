"""Conservative text normalization helpers."""

import html
import re
import unicodedata


_HTML_TAG_RE = re.compile(r"<\/?[a-zA-Z][^>]*>")
_HORIZONTAL_SPACE_RE = re.compile(r"[ \t\f\v]+")
_MULTIPLE_NEWLINES_RE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Remove obvious markup and normalize whitespace without rewriting prose."""

    if not text:
        return ""

    cleaned = (
        html.unescape(str(text))
        .replace("\u00a0", " ")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )
    cleaned = _HTML_TAG_RE.sub(" ", cleaned)
    lines = [_HORIZONTAL_SPACE_RE.sub(" ", line).strip() for line in cleaned.split("\n")]
    cleaned = "\n".join(lines)
    cleaned = _MULTIPLE_NEWLINES_RE.sub("\n\n", cleaned)
    return cleaned.strip()


def normalize_title(title: str) -> str:
    """Return a case- and punctuation-insensitive title form for deduplication."""

    normalized = clean_text(title).casefold()
    normalized = "".join(
        " " if unicodedata.category(character).startswith("P") else character
        for character in normalized
    )
    return re.sub(r"\s+", " ", normalized).strip()
