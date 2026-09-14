"""Small deterministic tokenizer for English news text."""

import re


_WORD_PATTERN = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase text and return word/number tokens."""

    if not text:
        return []
    return _WORD_PATTERN.findall(text.lower())
