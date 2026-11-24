"""Figures about what is indexed and how the chatbot has been doing.

    python -m docqa.stats

prints them. The chat screen shows the same figures in its side panel.

Two sources feed them: the document register and the query log, both in the SQLite
file, and the vector store for the number of passages of each kind. Nothing here
loads a model, so it is quick.

The times come from questions that were really asked on this machine. They are the
measurements to look at before changing a speed setting, and again afterwards.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass

from docqa.config import Config, ConfigError, load_config
from docqa.db import Database, DocumentRecord, ModelStats, QueryStats
from docqa.index.store import ChunkStore, StoreError

# Openings that most titles in a regulator's document set share. A list of names that
# all start with the same words is hard to scan, so the short name leaves them out.
_COMMON_OPENINGS = (
    re.compile(r"^reserve bank of india\b[\s,:–—-]*", re.IGNORECASE),
    re.compile(r"^master directions?\b[\s,:–—-]*(?:on\s+)?", re.IGNORECASE),
    re.compile(r"^master circulars?\b[\s,:–—-]*(?:on\s+)?", re.IGNORECASE),
)
_MIN_SHORT_LENGTH = 12


@dataclass(frozen=True)
class DocumentSummary:
    """One ingested document, as the chat screen lists it."""

    doc_id: str
    title: str
    short_title: str
    source_path: str
    pages: int | None
    passages: int
    effective_date: str | None
    version: str | None
    access_group: str
    ingested_at: str


@dataclass(frozen=True)
class IndexStats:
    """What is in the index."""

    documents: int = 0
    pages: int = 0
    passages: int = 0
    text_passages: int | None = None
    tables: int | None = None
    figures: int | None = None
    """The three counts by kind are ``None`` when the vector store was not consulted."""
    last_ingested_at: str | None = None


def short_title(title: str) -> str:
    """A shorter name for a document, for lists where the full title does not fit.

    Words that open most titles in the set ("Reserve Bank of India", "Master
    Direction -") are dropped, and so is a pair of brackets left dangling by that:
    "Reserve Bank of India (Small Finance Banks - Capital Adequacy) Directions, 2026"
    becomes "Small Finance Banks - Capital Adequacy, Directions, 2026". If too little
    would be left, the title is returned as it is.
    """
    full = " ".join(title.split())
    short = full
    for opening in _COMMON_OPENINGS:
        short = opening.sub("", short, count=1)
    short = short.replace("[", "(").replace("]", ")")  # a title that has not been through clean_title
    if short.startswith("("):
        close = _matching_bracket(short)
        if close == -1:
            short = short[1:]
        else:
            inside, rest = short[1:close].strip(), short[close + 1 :].strip(" ,")
            short = f"{inside}, {rest}" if rest else inside
    short = short.strip(" ,;:-–—")
    if len(short) < _MIN_SHORT_LENGTH:
        return full
    return short[0].upper() + short[1:]


def _matching_bracket(text: str) -> int:
    """Where the bracket that opens ``text`` is closed, or -1. Brackets inside it are skipped."""
    depth = 0
    for position, character in enumerate(text):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                return position
    return -1


def summarise_document(record: DocumentRecord) -> DocumentSummary:
    return DocumentSummary(
        doc_id=record.doc_id,
        title=record.doc_title,
        short_title=short_title(record.doc_title),
        source_path=record.source_path,
        pages=record.pages,
        passages=record.chunks,
        effective_date=record.effective_date.isoformat() if record.effective_date else None,
        version=record.version,
        access_group=record.access_group,
        ingested_at=record.ingested_at,
    )


def list_documents(db: Database) -> list[DocumentSummary]:
    """Every ingested document, in the order of its short name."""
    summaries = [summarise_document(record) for record in db.list_documents()]
    return sorted(summaries, key=lambda item: item.short_title.casefold())


def index_stats(db: Database, store: ChunkStore | None = None) -> IndexStats:
    """Totals over the document register, and passages by kind if a store is given."""
    records = db.list_documents()
    by_kind = store.count_by_type() if store is not None else None
    return IndexStats(
        documents=len(records),
        pages=sum(record.pages or 0 for record in records),
        passages=sum(record.chunks for record in records),
        text_passages=by_kind["text"] if by_kind else None,
        tables=by_kind["table"] if by_kind else None,
        figures=by_kind["figure"] if by_kind else None,
        last_ingested_at=max((record.ingested_at for record in records), default=None),
    )


def describe(index: IndexStats, queries: QueryStats, models: Sequence[ModelStats] = ()) -> list[str]:
    """The figures as lines of text. ``models``: the same, for each answer model on its own."""

    def seconds(value: float | None) -> str:
        if value is None:
            return "no data yet"
        if value < 0.1:
            return f"{value * 1000:.0f} ms"
        return f"{value:.1f} s" if value < 10 else f"{value:.0f} s"

    kinds = (
        f" ({index.text_passages} text, {index.tables} tables, {index.figures} figures)"
        if index.text_passages is not None
        else ""
    )
    speed = (
        "no data yet" if queries.median_tokens_per_s is None else f"{queries.median_tokens_per_s:.1f} tokens per second"
    )
    reading = (
        "no data yet"
        if queries.median_read_chars_per_s is None
        else f"{queries.median_read_chars_per_s:.0f} characters per second"
    )
    return [
        "Index",
        f"  documents:           {index.documents}",
        f"  pages:               {index.pages}",
        f"  passages:            {index.passages}{kinds}",
        f"  last ingested:       {index.last_ingested_at or 'never'}",
        "Questions",
        f"  asked:               {queries.questions} ({queries.answered} answered, "
        f"{queries.refused} refused, {queries.failed} failed)",
        f"  median total time:   {seconds(queries.median_latency_s)}",
        f"  median search time:  {seconds(queries.median_search_s)}",
        f"  of which database:   {seconds(queries.median_database_s)}",
        f"  of which ranking:    {seconds(queries.median_rerank_s)}",
        f"  median wait for the first word: {seconds(queries.median_first_token_s)}",
        f"  median reading speed: {reading}",
        f"  median writing speed: {speed}",
        f"  last asked:          {queries.last_asked_at or 'never'}",
        *_by_model(models, seconds),
    ]


def _by_model(models: Sequence[ModelStats], seconds) -> list[str]:
    """One line per answer model, so that two models can be compared after a change."""
    if not models:
        return []
    lines = ["By answer model"]
    for item in models:
        parts = [f"{item.answers} answer(s)", f"median total {seconds(item.median_latency_s)}"]
        if item.median_first_token_s is not None:
            parts.append(f"first word after {seconds(item.median_first_token_s)}")
        if item.median_read_chars_per_s is not None:
            parts.append(f"reads {item.median_read_chars_per_s:.0f} characters/s")
        if item.median_tokens_per_s is not None:
            parts.append(f"writes {item.median_tokens_per_s:.1f} tokens/s")
        lines.append(f"  {item.answer_model}: " + ", ".join(parts))
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m docqa.stats",
        description="Show what is indexed and how long questions have been taking.",
    )
    parser.add_argument("--config", help="path to config.yaml (default: the project's own)")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(exc)
        return 1
    from docqa import resources

    db = resources.get_database(config)
    try:
        store: ChunkStore | None = resources.get_store(config)
    except StoreError as exc:
        print(f"(Passages by kind are left out: {exc})\n")
        store = None
    try:
        print("\n".join(describe(index_stats(db, store), db.query_stats(), db.model_stats())))
    finally:
        resources.close_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
