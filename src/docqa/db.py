"""SQLite storage: the document register, the figure-description cache and the query log.

One file, ``paths.sqlite_path``. Three tables:

``documents``
    One row per ingested file: its hash, the settings it was ingested with, and the
    title, version, effective date and access group its chunks carry. Ingestion reads
    it to decide whether a file can be skipped.

``figure_descriptions``
    What the vision model wrote about a figure, keyed by the hash of the cropped image,
    the model name and the version of the instruction. The vision model is the slowest
    step in the pipeline, so nothing is ever described twice.

``query_log``
    One row per question asked: time, question, the chunks retrieved with their scores,
    the answer, how long it took and which models were used.

Every method opens its own connection and closes it again. That is slower than keeping
one open, and it does not matter at this scale; in exchange the class is safe to use
from the chat screen's server, which answers each request on a worker thread.
"""

from __future__ import annotations

import json
import sqlite3
import statistics
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from docqa.schema import Answer, FigureDescription, FlowchartEdge

_TABLES = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id          TEXT PRIMARY KEY,
    source_path     TEXT NOT NULL,
    sha256          TEXT NOT NULL,
    fingerprint     TEXT NOT NULL,
    doc_title       TEXT NOT NULL,
    version         TEXT,
    effective_date  TEXT,
    access_group    TEXT NOT NULL,
    pages           INTEGER,
    chunks          INTEGER NOT NULL,
    ingested_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS figure_descriptions (
    image_sha256    TEXT NOT NULL,
    model           TEXT NOT NULL,
    prompt_version  TEXT NOT NULL,
    description     TEXT NOT NULL,
    raw_reply       TEXT NOT NULL,
    latency_s       REAL,
    created_at      TEXT NOT NULL,
    PRIMARY KEY (image_sha256, model, prompt_version)
);

CREATE TABLE IF NOT EXISTS query_log (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    asked_at         TEXT NOT NULL,
    question         TEXT NOT NULL,
    answer           TEXT NOT NULL,
    refused          INTEGER NOT NULL,
    error            TEXT,
    retrieved        TEXT NOT NULL,
    citations        TEXT NOT NULL,
    latency_s        REAL NOT NULL,
    answer_model     TEXT,
    embedding_model  TEXT NOT NULL,
    reranker_model   TEXT NOT NULL,
    timings          TEXT
);
"""

# Columns added after the first version. A database made by an older version of the
# code gets them added when it is opened, so that its rows are kept.
_ADDED_COLUMNS = (("query_log", "timings", "TEXT"),)


@dataclass(frozen=True)
class DocumentRecord:
    """One row of the document register."""

    doc_id: str
    source_path: str
    """Path as it was given to ingestion, relative to ``paths.docs_dir`` where possible."""
    sha256: str
    fingerprint: str
    """Hash of everything other than the file that decides what its chunks look like:
    the document's metadata and the parsing, chunking and embedding settings. A file is
    skipped only if both ``sha256`` and ``fingerprint`` are unchanged."""
    doc_title: str
    version: str | None
    effective_date: date | None
    access_group: str
    pages: int | None
    chunks: int
    ingested_at: str
    """UTC time, ISO 8601."""


@dataclass(frozen=True)
class LoggedQuery:
    """One row of the query log, as read back."""

    id: int
    asked_at: str
    question: str
    answer: str
    refused: bool
    error: str | None
    retrieved: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    latency_s: float
    answer_model: str | None
    embedding_model: str
    reranker_model: str
    timings: dict[str, Any] | None = None
    """Where the time went (``AnswerTimings`` as plain data). ``None`` for rows logged
    before timings were recorded."""


@dataclass(frozen=True)
class QueryStats:
    """Figures drawn from the query log, for the chat screen's side panel."""

    questions: int = 0
    answered: int = 0
    refused: int = 0
    failed: int = 0
    median_latency_s: float | None = None
    """Median seconds from question to finished answer, over answered questions."""
    median_search_s: float | None = None
    """Median seconds spent searching and reranking."""
    median_database_s: float | None = None
    """Median seconds of that which the vector store took. The rest is the two search models."""
    median_first_token_s: float | None = None
    """Median seconds the answer model took before its first word."""
    median_tokens_per_s: float | None = None
    """Median writing speed of the answer model, in tokens per second."""
    last_asked_at: str | None = None


def utc_now() -> str:
    """The current UTC time as text, to the second."""
    return datetime.now(UTC).isoformat(timespec="seconds")


class Database:
    """The project's SQLite file. Creating the object creates the file and its tables."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(_TABLES)
            for table, column, kind in _ADDED_COLUMNS:
                present = {row["name"] for row in connection.execute(f"PRAGMA table_info({table})")}
                if column not in present:
                    connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            with connection:  # commits on success, rolls back on an exception
                yield connection
        finally:
            connection.close()

    # -- document register ------------------------------------------------------------

    def get_document(self, doc_id: str) -> DocumentRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
        return _document(row) if row else None

    def list_documents(self) -> list[DocumentRecord]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM documents ORDER BY source_path").fetchall()
        return [_document(row) for row in rows]

    def save_document(self, record: DocumentRecord) -> None:
        """Insert the record, or replace the one with the same ``doc_id``."""
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.doc_id,
                    record.source_path,
                    record.sha256,
                    record.fingerprint,
                    record.doc_title,
                    record.version,
                    record.effective_date.isoformat() if record.effective_date else None,
                    record.access_group,
                    record.pages,
                    record.chunks,
                    record.ingested_at,
                ),
            )

    def delete_document(self, doc_id: str) -> bool:
        """Remove a document from the register. True if it was there."""
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            return cursor.rowcount > 0

    # -- figure description cache -----------------------------------------------------

    def get_figure_description(
        self, image_sha256: str, model: str, prompt_version: str
    ) -> FigureDescription | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT description FROM figure_descriptions "
                "WHERE image_sha256 = ? AND model = ? AND prompt_version = ?",
                (image_sha256, model, prompt_version),
            ).fetchone()
        return description_from_json(row["description"]) if row else None

    def save_figure_description(
        self,
        description: FigureDescription,
        prompt_version: str,
        raw_reply: str,
        latency_s: float | None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO figure_descriptions VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    description.image_sha256,
                    description.model,
                    prompt_version,
                    description_to_json(description),
                    raw_reply,
                    latency_s,
                    utc_now(),
                ),
            )

    def count_figure_descriptions(self) -> int:
        with self._connect() as connection:
            return connection.execute("SELECT COUNT(*) FROM figure_descriptions").fetchone()[0]

    # -- query log --------------------------------------------------------------------

    def log_answer(self, answer: Answer, embedding_model: str, reranker_model: str) -> int:
        """Record one answered (or refused) question. Returns the id of the new row."""
        return self._log(
            question=answer.question,
            answer=answer.text,
            refused=answer.refused,
            error=None,
            retrieved=_retrieved(answer),
            citations=[asdict(citation) for citation in answer.citations],
            latency_s=answer.latency_s,
            answer_model=answer.answer_model,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
            timings=asdict(answer.timings),
        )

    def log_failure(
        self,
        question: str,
        error: str,
        latency_s: float,
        embedding_model: str,
        reranker_model: str,
        answer_model: str | None = None,
    ) -> int:
        """Record a question that could not be answered because something failed."""
        return self._log(
            question=question,
            answer="",
            refused=False,
            error=error,
            retrieved=[],
            citations=[],
            latency_s=latency_s,
            answer_model=answer_model,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
        )

    def recent_queries(self, limit: int = 20) -> list[LoggedQuery]:
        """The newest rows of the query log, newest first."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM query_log ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [
            LoggedQuery(
                id=row["id"],
                asked_at=row["asked_at"],
                question=row["question"],
                answer=row["answer"],
                refused=bool(row["refused"]),
                error=row["error"],
                retrieved=json.loads(row["retrieved"]),
                citations=json.loads(row["citations"]),
                latency_s=row["latency_s"],
                answer_model=row["answer_model"],
                embedding_model=row["embedding_model"],
                reranker_model=row["reranker_model"],
                timings=json.loads(row["timings"]) if row["timings"] else None,
            )
            for row in rows
        ]

    def query_stats(self) -> QueryStats:
        """Counts and typical times over the whole query log."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT asked_at, refused, error, latency_s, timings FROM query_log ORDER BY id"
            ).fetchall()
        if not rows:
            return QueryStats()

        answered = [row for row in rows if not row["error"] and not row["refused"]]
        search, database, first_token, speed = [], [], [], []
        for row in rows:
            timings = json.loads(row["timings"]) if row["timings"] else None
            if not timings or row["error"]:
                continue
            retrieval = timings.get("retrieval") or {}
            spent = sum(retrieval.get(part) or 0.0 for part in ("embed_s", "search_s", "rerank_s"))
            if spent:
                search.append(spent)
                database.append(retrieval.get("search_s") or 0.0)
            if row["refused"]:
                continue
            if timings.get("first_token_s") is not None:
                first_token.append(timings["first_token_s"])
            writing = (timings.get("generation_s") or 0.0) - (timings.get("first_token_s") or 0.0)
            if timings.get("completion_tokens") and writing > 0:
                speed.append(timings["completion_tokens"] / writing)
        return QueryStats(
            questions=len(rows),
            answered=len(answered),
            refused=sum(1 for row in rows if row["refused"] and not row["error"]),
            failed=sum(1 for row in rows if row["error"]),
            median_latency_s=_median([row["latency_s"] for row in answered]),
            median_search_s=_median(search),
            median_database_s=_median(database),
            median_first_token_s=_median(first_token),
            median_tokens_per_s=_median(speed),
            last_asked_at=rows[-1]["asked_at"],
        )

    def _log(
        self,
        *,
        question: str,
        answer: str,
        refused: bool,
        error: str | None,
        retrieved: Sequence[dict[str, Any]],
        citations: Sequence[dict[str, Any]],
        latency_s: float,
        answer_model: str | None,
        embedding_model: str,
        reranker_model: str,
        timings: dict[str, Any] | None = None,
    ) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO query_log (asked_at, question, answer, refused, error, retrieved, "
                "citations, latency_s, answer_model, embedding_model, reranker_model, timings) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    utc_now(),
                    question,
                    answer,
                    int(refused),
                    error,
                    json.dumps(list(retrieved), ensure_ascii=False),
                    json.dumps(list(citations), ensure_ascii=False),
                    latency_s,
                    answer_model,
                    embedding_model,
                    reranker_model,
                    json.dumps(timings) if timings is not None else None,
                ),
            )
            return int(cursor.lastrowid or 0)


def description_to_json(description: FigureDescription) -> str:
    """A figure description as JSON text."""
    return json.dumps(asdict(description), ensure_ascii=False)


def description_from_json(text: str) -> FigureDescription:
    """The reverse of ``description_to_json``."""
    data = json.loads(text)
    return FigureDescription(
        image_sha256=data["image_sha256"],
        model=data["model"],
        figure_type=data["figure_type"],
        title=data.get("title"),
        axes_and_legend=data.get("axes_and_legend"),
        readable_values=tuple(data.get("readable_values") or ()),
        nodes=tuple(data.get("nodes") or ()),
        edges=tuple(
            FlowchartEdge(edge["source"], edge["target"], edge.get("condition"))
            for edge in data.get("edges") or ()
        ),
        raw_text=data.get("raw_text"),
    )


def _document(row: sqlite3.Row) -> DocumentRecord:
    return DocumentRecord(
        doc_id=row["doc_id"],
        source_path=row["source_path"],
        sha256=row["sha256"],
        fingerprint=row["fingerprint"],
        doc_title=row["doc_title"],
        version=row["version"],
        effective_date=date.fromisoformat(row["effective_date"]) if row["effective_date"] else None,
        access_group=row["access_group"],
        pages=row["pages"],
        chunks=row["chunks"],
        ingested_at=row["ingested_at"],
    )


def _retrieved(answer: Answer) -> list[dict[str, Any]]:
    given = set(answer.evidence_ids)
    return [
        {
            "chunk_id": source.chunk.chunk_id,
            "doc_id": source.chunk.doc_id,
            "doc_title": source.chunk.doc_title,
            "page": source.chunk.page,
            "element_type": source.chunk.element_type.value,
            "fusion_score": source.fusion_score,
            "rerank_score": source.rerank_score,
            "given_to_model": source.chunk.chunk_id in given,
        }
        for source in answer.sources
    ]


def _median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None
