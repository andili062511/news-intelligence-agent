"""Load news chunks from JSON Lines files."""

import json
from pathlib import Path
from typing import Any


def load_chunks(path: str | Path = "data/chunks.jsonl") -> list[dict[str, Any]]:
    """Read one chunk object per line from a UTF-8 JSONL file."""

    chunks_path = Path(path)
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")
    if not chunks_path.is_file():
        raise FileNotFoundError(f"Chunks path is not a file: {chunks_path}")

    chunks: list[dict[str, Any]] = []
    with chunks_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {chunks_path} on line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(chunk, dict):
                raise ValueError(
                    f"Expected a JSON object in {chunks_path} on line {line_number}"
                )
            chunks.append(chunk)

    return chunks
