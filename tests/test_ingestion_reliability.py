import json
from pathlib import Path

import pytest

from ingestion import pipeline


def _write_existing_dataset(output_dir):
    articles_content = '{"article_id": "existing"}\n'
    chunks_content = '{"chunk_id": "existing_0"}\n'
    (output_dir / "articles.jsonl").write_text(articles_content, encoding="utf-8")
    (output_dir / "chunks.jsonl").write_text(chunks_content, encoding="utf-8")
    return articles_content, chunks_content


def test_empty_fetch_does_not_overwrite_articles(monkeypatch, tmp_path, capsys):
    articles_content, _ = _write_existing_dataset(tmp_path)
    monkeypatch.setattr(pipeline, "fetch_news", lambda *args, **kwargs: ([], "GDELT"))

    with pytest.raises(pipeline.IngestionError):
        pipeline.run_pipeline("test", output_dir=tmp_path)

    assert (tmp_path / "articles.jsonl").read_text(encoding="utf-8") == articles_content
    assert pipeline.NO_ARTICLES_WARNING in capsys.readouterr().out


def test_empty_fetch_does_not_overwrite_chunks(monkeypatch, tmp_path):
    _, chunks_content = _write_existing_dataset(tmp_path)
    monkeypatch.setattr(pipeline, "fetch_news", lambda *args, **kwargs: ([], "GDELT"))

    with pytest.raises(pipeline.IngestionError):
        pipeline.run_pipeline("test", output_dir=tmp_path)

    assert (tmp_path / "chunks.jsonl").read_text(encoding="utf-8") == chunks_content


def test_zero_valid_articles_preserves_existing_dataset(monkeypatch, tmp_path):
    articles_content, chunks_content = _write_existing_dataset(tmp_path)
    monkeypatch.setattr(
        pipeline,
        "fetch_news",
        lambda *args, **kwargs: ([
            {"title": "Turkish article", "language": "Turkish"}
        ], "GDELT"),
    )

    with pytest.raises(pipeline.IngestionError):
        pipeline.run_pipeline("test", language="English", output_dir=tmp_path)

    assert (tmp_path / "articles.jsonl").read_text(encoding="utf-8") == articles_content
    assert (tmp_path / "chunks.jsonl").read_text(encoding="utf-8") == chunks_content
    assert not list(tmp_path.glob("*.tmp"))


def test_successful_pipeline_replaces_both_dataset_files(monkeypatch, tmp_path):
    articles_content, chunks_content = _write_existing_dataset(tmp_path)
    monkeypatch.setattr(
        pipeline,
        "fetch_news",
        lambda *args, **kwargs: ([
            {"title": "New article", "summary": "New content", "language": "English"}
        ], "GDELT"),
    )

    articles, chunks = pipeline.run_pipeline("test", output_dir=tmp_path)

    new_articles_content = (tmp_path / "articles.jsonl").read_text(encoding="utf-8")
    new_chunks_content = (tmp_path / "chunks.jsonl").read_text(encoding="utf-8")
    assert new_articles_content != articles_content
    assert new_chunks_content != chunks_content
    assert [json.loads(line) for line in new_articles_content.splitlines()] == articles
    assert [json.loads(line) for line in new_chunks_content.splitlines()] == chunks
    assert not (tmp_path / "articles.jsonl.tmp").exists()
    assert not (tmp_path / "chunks.jsonl.tmp").exists()


def test_temporary_files_are_cleaned_when_staging_fails(monkeypatch, tmp_path):
    articles_content, chunks_content = _write_existing_dataset(tmp_path)
    original_write_jsonl = pipeline._write_jsonl

    def fail_while_writing_chunks(path, records):
        if path.name == "chunks.jsonl.tmp":
            raise OSError("simulated write failure")
        original_write_jsonl(path, records)

    monkeypatch.setattr(pipeline, "_write_jsonl", fail_while_writing_chunks)

    with pytest.raises(OSError, match="simulated write failure"):
        pipeline._publish_dataset(tmp_path, [{"article_id": "new"}], [{"chunk_id": "new_0"}])

    assert (tmp_path / "articles.jsonl").read_text(encoding="utf-8") == articles_content
    assert (tmp_path / "chunks.jsonl").read_text(encoding="utf-8") == chunks_content
    assert not (tmp_path / "articles.jsonl.tmp").exists()
    assert not (tmp_path / "chunks.jsonl.tmp").exists()


def test_second_atomic_replace_failure_rolls_back_first_file(monkeypatch, tmp_path):
    articles_content, chunks_content = _write_existing_dataset(tmp_path)
    original_replace = pipeline.os.replace

    def fail_replacing_chunks(source, destination):
        if Path(source).name == "chunks.jsonl.tmp":
            raise OSError("simulated replace failure")
        original_replace(source, destination)

    monkeypatch.setattr(pipeline.os, "replace", fail_replacing_chunks)

    with pytest.raises(OSError, match="simulated replace failure"):
        pipeline._publish_dataset(
            tmp_path,
            [{"article_id": "new"}],
            [{"chunk_id": "new_0"}],
        )

    assert (tmp_path / "articles.jsonl").read_text(encoding="utf-8") == articles_content
    assert (tmp_path / "chunks.jsonl").read_text(encoding="utf-8") == chunks_content
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob("*.backup"))


def test_publish_rejects_empty_records_and_cleans_stale_temporary_files(tmp_path):
    articles_content, chunks_content = _write_existing_dataset(tmp_path)
    (tmp_path / "articles.jsonl.tmp").write_text("stale", encoding="utf-8")
    (tmp_path / "chunks.jsonl.tmp").write_text("stale", encoding="utf-8")

    with pytest.raises(pipeline.IngestionError, match="empty dataset"):
        pipeline._publish_dataset(tmp_path, [], [])

    assert (tmp_path / "articles.jsonl").read_text(encoding="utf-8") == articles_content
    assert (tmp_path / "chunks.jsonl").read_text(encoding="utf-8") == chunks_content
    assert not list(tmp_path.glob("*.tmp"))


def test_cli_returns_failure_status_for_empty_fetch(monkeypatch):
    monkeypatch.setattr(pipeline.sys, "argv", ["ingestion.pipeline", "--query", "test"])
    monkeypatch.setattr(pipeline, "fetch_news", lambda *args, **kwargs: ([], "GDELT"))

    assert pipeline.main() == 1
