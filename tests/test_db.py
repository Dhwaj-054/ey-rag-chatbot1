"""The SQLite file: the document register and the query log."""

from __future__ import annotations

import sqlite3
import threading
from datetime import date

from docqa.db import Database, DocumentRecord, utc_now


def record(doc_id: str = "doc1", **changes) -> DocumentRecord:
    values = {
        "doc_id": doc_id,
        "source_path": f"{doc_id}.pdf",
        "sha256": "a" * 64,
        "fingerprint": "f" * 64,
        "doc_title": "Lending Policy",
        "version": "v2",
        "effective_date": date(2026, 4, 1),
        "access_group": "all",
        "pages": 14,
        "chunks": 31,
        "ingested_at": utc_now(),
    }
    values.update(changes)
    return DocumentRecord(**values)


def test_creating_the_database_creates_the_file_its_folder_and_its_tables(tmp_path):
    path = tmp_path / "nested" / "data" / "docqa.sqlite"

    Database(path)
    Database(path)  # opening an existing file is fine

    with sqlite3.connect(path) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"documents", "figure_descriptions", "query_log"} <= tables


def test_a_document_record_is_saved_read_replaced_and_deleted(tmp_path):
    database = Database(tmp_path / "docqa.sqlite")
    assert database.get_document("doc1") is None

    first = record()
    database.save_document(first)
    assert database.get_document("doc1") == first

    changed = record(sha256="b" * 64, chunks=40, effective_date=None, version=None)
    database.save_document(changed)
    assert database.get_document("doc1") == changed
    assert len(database.list_documents()) == 1

    assert database.delete_document("doc1") is True
    assert database.delete_document("doc1") is False
    assert database.list_documents() == []


def test_documents_are_listed_in_path_order(tmp_path):
    database = Database(tmp_path / "docqa.sqlite")
    for doc_id in ("zeta", "alpha", "mid"):
        database.save_document(record(doc_id))

    assert [item.doc_id for item in database.list_documents()] == ["alpha", "mid", "zeta"]


def test_failed_questions_are_logged_too(tmp_path):
    database = Database(tmp_path / "docqa.sqlite")

    row_id = database.log_failure(
        "What is the fee?", "LLMConnectionError: cannot reach the server", 3.2, "BAAI/bge-m3", "reranker", "qwen2.5:7b"
    )

    logged = database.recent_queries()[0]
    assert logged.id == row_id
    assert (logged.question, logged.answer, logged.refused) == ("What is the fee?", "", False)
    assert logged.error == "LLMConnectionError: cannot reach the server"
    assert logged.latency_s == 3.2
    assert logged.answer_model == "qwen2.5:7b"


def test_the_database_can_be_used_from_several_threads(tmp_path):
    """Streamlit runs each page refresh on another thread."""
    database = Database(tmp_path / "docqa.sqlite")
    errors: list[Exception] = []

    def work(number: int) -> None:
        try:
            database.save_document(record(f"doc{number}"))
            database.log_failure(f"question {number}", "error", 0.1, "e", "r")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(number,)) for number in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(database.list_documents()) == 8
    assert len(database.recent_queries(50)) == 8
