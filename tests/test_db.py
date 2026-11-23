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


def test_a_database_made_by_the_first_version_is_upgraded_and_keeps_its_rows(tmp_path):
    """The query log gained a column for timings. Rows logged before that must survive."""
    path = tmp_path / "docqa.sqlite"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE query_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT, asked_at TEXT NOT NULL, question TEXT NOT NULL,
                answer TEXT NOT NULL, refused INTEGER NOT NULL, error TEXT, retrieved TEXT NOT NULL,
                citations TEXT NOT NULL, latency_s REAL NOT NULL, answer_model TEXT,
                embedding_model TEXT NOT NULL, reranker_model TEXT NOT NULL
            );
            INSERT INTO query_log (asked_at, question, answer, refused, error, retrieved, citations,
                                   latency_s, answer_model, embedding_model, reranker_model)
            VALUES ('2026-11-21T10:00:00+00:00', 'An old question', 'An old answer', 0, NULL, '[]', '[]',
                    329.0, 'qwen2.5:7b', 'BAAI/bge-m3', 'BAAI/bge-reranker-v2-m3');
            """
        )

    database = Database(path)
    Database(path)  # opening it a second time changes nothing
    database.log_failure("A new question", "error", 1.0, "e", "r")

    newest, oldest = database.recent_queries()
    assert (oldest.question, oldest.answer, oldest.latency_s, oldest.timings) == (
        "An old question",
        "An old answer",
        329.0,
        None,
    )
    assert newest.question == "A new question"
    stats = database.query_stats()
    assert (stats.questions, stats.answered, stats.failed) == (2, 1, 1)
    # The old row has no timings, so it counts towards the total time only.
    assert stats.median_latency_s == 329.0 and stats.median_search_s is None


def test_the_database_can_be_used_from_several_threads(tmp_path):
    """The chat screen's server answers each request on another thread."""
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


def test_a_document_can_be_found_by_the_content_of_its_file(tmp_path):
    """How the chat screen notices that an uploaded PDF is ingested already."""
    database = Database(tmp_path / "docqa.sqlite")
    database.save_document(record("doc1", sha256="a" * 64, source_path="Circulars/a.pdf"))
    database.save_document(record("doc2", sha256="b" * 64, source_path="b.pdf"))
    database.save_document(record("doc3", sha256="b" * 64, source_path="a copy of b.pdf"))

    assert database.find_document_by_hash("a" * 64).doc_id == "doc1"
    assert database.find_document_by_hash("b" * 64).source_path == "a copy of b.pdf"  # the first by path
    assert database.find_document_by_hash("c" * 64) is None
