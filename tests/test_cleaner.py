from ingestion.cleaner import clean_text, normalize_title


def test_clean_text_normalizes_markup_and_whitespace() -> None:
    text = "  Hello&nbsp;world\r\n\r\n\r\n <p>News</p>   today  "
    assert clean_text(text) == "Hello world\n\nNews today"


def test_normalize_title_is_case_and_punctuation_insensitive() -> None:
    assert normalize_title("OpenAI Launches New Model!!!") == "openai launches new model"
    assert normalize_title(" openai launches new model ") == "openai launches new model"

