from retrieval.tokenizer import tokenize


def test_tokenize_lowercases_text() -> None:
    assert tokenize("OpenAI Launches GPT") == ["openai", "launches", "gpt"]


def test_tokenize_removes_punctuation() -> None:
    assert tokenize("Hello, world! GPT-5.6.") == ["hello", "world", "gpt", "5", "6"]


def test_tokenize_handles_empty_string() -> None:
    assert tokenize("") == []


def test_tokenize_keeps_numbers() -> None:
    assert tokenize("In 2026, version 5.6 launched") == [
        "in",
        "2026",
        "version",
        "5",
        "6",
        "launched",
    ]
