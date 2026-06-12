from src.io_utils import extra_fields_from_row, iter_document_inputs


def test_iter_document_inputs_preserves_extra_columns():
    rows = [
        {
            "id": "p1",
            "text": "hello",
            "source": "bench",
            "category": "harmful",
            "severity": 3,
            "tags": ["a", "b"],
            "meta": {"k": 1},
        }
    ]
    doc_id, text, meta = next(iter(iter_document_inputs(rows, "text", "id")))
    assert doc_id == "p1"
    assert text == "hello"
    assert meta["source"] == "bench"
    assert meta["category"] == "harmful"
    assert meta["extra"] == {"severity": 3, "tags": ["a", "b"], "meta": {"k": 1}}


def test_iter_document_inputs_excludes_text_and_id_columns():
    rows = [{"uid": "u1", "body": "x", "note": "n"}]
    _, text, meta = next(iter(iter_document_inputs(rows, "body", "uid")))
    assert text == "x"
    assert meta["extra"] == {"note": "n"}


def test_extra_fields_from_row_helper():
    row = {"id": 1, "text": "t", "source": "s", "category": "c", "x": 1}
    assert extra_fields_from_row(row, "text", "id") == {"x": 1}
