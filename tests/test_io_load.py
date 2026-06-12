import pytest

from src.io_utils import (
    iter_document_inputs,
    load_documents,
    validate_input_file,
)


def test_load_jsonl_ok(tmp_path):
    p = tmp_path / "corpus.jsonl"
    p.write_text(
        '{"id": "1", "text": "hello", "tag": "a"}\n{"id": "2", "text": "world"}\n',
        encoding="utf-8",
    )
    rows = load_documents(p)
    assert len(rows) == 2
    assert rows[0]["text"] == "hello"
    assert rows[0]["tag"] == "a"


def test_load_jsonl_missing_text_field_raises(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"id": "1"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        load_documents(p)


def test_load_jsonl_invalid_json_raises(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"id":"1","text":"ok"}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_documents(p)


def test_load_jsonl_skip_malformed_lines(tmp_path):
    p = tmp_path / "mixed.jsonl"
    p.write_text(
        '{"id":"1","text":"a"}\n'
        "not json\n"
        '{"id":"2","text":"b"}\n',
        encoding="utf-8",
    )
    rows = load_documents(p, skip_malformed_jsonl_lines=True)
    assert len(rows) == 2
    assert [r["id"] for r in rows] == ["1", "2"]


def test_load_csv_ok(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("id,text,extra\n1,hello,x\n", encoding="utf-8")
    rows = load_documents(p, input_format="csv")
    assert len(rows) == 1
    assert rows[0]["text"] == "hello"
    assert rows[0]["extra"] == "x"


def test_load_csv_missing_text_column_raises(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("id,body\n1,hello\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no column"):
        load_documents(p, text_column="text", input_format="csv")


def test_validate_input_file_missing(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        validate_input_file(tmp_path / "does_not_exist.jsonl")


def test_iter_document_inputs_empty_text_preserved():
    rows = [{"id": "z", "text": ""}]
    doc_id, text, meta = next(iter(iter_document_inputs(rows, "text", "id")))
    assert doc_id == "z"
    assert text == ""
    assert meta["extra"] == {}
