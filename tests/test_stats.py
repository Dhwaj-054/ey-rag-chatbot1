"""The figures shown in the chat screen's side panel, and short names for documents."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import pytest

from docqa.db import Database, DocumentRecord, QueryStats
from docqa.index import index_chunks
from docqa.schema import (
    Answer,
    AnswerTimings,
    ElementType,
    RetrievalTimings,
    ScoredChunk,
)
from docqa.stats import IndexStats, describe, index_stats, list_documents, main, short_title
from tests.fakes import FakeEmbedder, make_chunk

# -- short names ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("title", "short"),
    [
        (
            "Reserve Bank of India (Small Finance Banks – Prudential Norms on Capital Adequacy) Sixth Amendment Directions, 2026",
            "Small Finance Banks – Prudential Norms on Capital Adequacy, Sixth Amendment Directions, 2026",
        ),
        (
            "Reserve Bank of India (Domestic Money Transfer) Directions, 2026",
            "Domestic Money Transfer, Directions, 2026",
        ),
        (
            "Master Direction – Operational Guidelines for Primary Dealers (Updated as on November 2, 2026)",
            "Operational Guidelines for Primary Dealers (Updated as on November 2, 2026)",
        ),
        ("Master Directions on Relief/Savings Bonds", "Relief/Savings Bonds"),
        (
            "Reserve Bank of India (Commercial Banks - Kisan Credit Card (KCC) Scheme) Directions, 2026",
            "Commercial Banks - Kisan Credit Card (KCC) Scheme, Directions, 2026",
        ),
        (
            "Reserve Bank of India [Conduct of Government Business by Agency Banks (ABs)] Directions, 2026",
            "Conduct of Government Business by Agency Banks (ABs), Directions, 2026",
        ),
        ("Reserve Bank of India (Unfinished bracket", "Unfinished bracket"),
        (
            "Master Circular - Credit Facilities to Scheduled Castes (SCs) & Scheduled Tribes (STs)",
            "Credit Facilities to Scheduled Castes (SCs) & Scheduled Tribes (STs)",
        ),
        (
            "Penal Interest on shortfall in CRR and SLR requirements - Change in Bank Rate",
            "Penal Interest on shortfall in CRR and SLR requirements - Change in Bank Rate",
        ),
        ("Notice on Media Policy- Advisory Notice", "Notice on Media Policy- Advisory Notice"),
    ],
)
def test_short_names_drop_the_words_that_every_title_starts_with(title, short):
    assert short_title(title) == short


def test_a_title_that_would_become_too_short_is_kept_whole():
    assert short_title("Reserve Bank of India") == "Reserve Bank of India"
    assert short_title("Master Circular - KYC") == "Master Circular - KYC"
    assert short_title("  Reserve   Bank of India  (Act)  ") == "Reserve Bank of India (Act)"


# -- the index --------------------------------------------------------------------------


def register(db: Database, doc_id: str, title: str, pages: int, chunks: int, when=None, ingested="2026-11-20T10:00:00+00:00"):
    db.save_document(
        DocumentRecord(doc_id, f"{doc_id}.pdf", "0" * 64, "0" * 64, title, "v1", when, "all", pages, chunks, ingested)
    )


def test_documents_are_listed_by_short_name_with_their_full_title(tmp_cfg):
    db = Database(tmp_cfg.paths.sqlite_path)
    register(db, "kyc", "Reserve Bank of India (Know Your Customer) Directions, 2026", 80, 120, date(2026, 5, 1))
    register(db, "bonds", "Master Directions on Relief/Savings Bonds", 12, 9)

    documents = list_documents(db)

    assert [doc.short_title for doc in documents] == ["Know Your Customer, Directions, 2026", "Relief/Savings Bonds"]
    kyc = documents[0]
    assert kyc.title == "Reserve Bank of India (Know Your Customer) Directions, 2026"
    assert (kyc.pages, kyc.passages, kyc.effective_date, kyc.version) == (80, 120, "2026-05-01", "v1")
    assert kyc.source_path == "kyc.pdf"


def test_index_totals_come_from_the_register_and_kinds_from_the_store(tmp_cfg, store):
    db = Database(tmp_cfg.paths.sqlite_path)
    assert index_stats(db) == IndexStats()

    register(db, "a", "Document A", 10, 3, ingested="2026-11-20T10:00:00+00:00")
    register(db, "b", "Document B", 5, 1, ingested="2026-11-21T09:30:00+00:00")
    index_chunks(
        [
            make_chunk("a:00000", "Some text."),
            make_chunk("a:00001", "Table: T", element_type=ElementType.TABLE),
            make_chunk("a:00002", "Figure (bar chart)", element_type=ElementType.FIGURE),
            make_chunk("b:00000", "More text."),
        ],
        tmp_cfg,
        embedder=FakeEmbedder(),
        store=store,
    )

    without_store = index_stats(db)
    with_store = index_stats(db, store)

    assert (without_store.documents, without_store.pages, without_store.passages) == (2, 15, 4)
    assert without_store.text_passages is None
    assert (with_store.text_passages, with_store.tables, with_store.figures) == (2, 1, 1)
    assert with_store.last_ingested_at == "2026-11-21T09:30:00+00:00"


# -- the questions ------------------------------------------------------------------------


def logged_answer(latency: float, rerank: float, first_token: float, generation: float, tokens: int, refused=False) -> Answer:
    source = ScoredChunk(make_chunk(), 0.03, 0.9)
    return Answer(
        question="q",
        text="I could not find this in the documents." if refused else "An answer.",
        citations=(),
        sources=(source,),
        refused=refused,
        answer_model=None if refused else "qwen2.5:7b",
        latency_s=latency,
        timings=AnswerTimings(
            retrieval=RetrievalTimings(embed_s=1.0, search_s=0.5, rerank_s=rerank, candidates=30),
            first_token_s=None if refused else first_token,
            generation_s=0.0 if refused else generation,
            completion_tokens=None if refused else tokens,
        ),
    )


def test_query_statistics_are_medians_over_the_log(tmp_cfg):
    db = Database(tmp_cfg.paths.sqlite_path)
    assert db.query_stats() == QueryStats()

    db.log_answer(logged_answer(300, rerank=78.5, first_token=180, generation=220, tokens=200), "e", "r")
    db.log_answer(logged_answer(100, rerank=28.5, first_token=50, generation=70, tokens=120), "e", "r")
    db.log_answer(logged_answer(200, rerank=48.5, first_token=110, generation=150, tokens=240), "e", "r")
    db.log_answer(logged_answer(40, rerank=38.5, first_token=0, generation=0, tokens=0, refused=True), "e", "r")
    db.log_failure("broken", "LLMConnectionError: down", 5.0, "e", "r")

    stats = db.query_stats()

    assert (stats.questions, stats.answered, stats.refused, stats.failed) == (5, 3, 1, 1)
    assert stats.median_latency_s == 200  # answered questions only: 100, 200, 300
    assert stats.median_search_s == pytest.approx(45.0)  # 30, 40, 50, 80 seconds of search
    assert stats.median_database_s == 0.5  # the vector store's share of each of them
    assert stats.median_first_token_s == 110
    assert stats.median_tokens_per_s == pytest.approx(6.0)  # 5, 6 and 6 tokens per second
    assert stats.last_asked_at is not None


def test_the_figures_read_as_plain_lines(tmp_cfg):
    index = IndexStats(documents=15, pages=62, passages=79, text_passages=70, tables=6, figures=3, last_ingested_at="2026-11-21")
    queries = QueryStats(
        questions=4,
        answered=3,
        refused=1,
        median_latency_s=329.2,
        median_search_s=81.0,
        median_database_s=0.0062,
        median_tokens_per_s=6.04,
    )

    lines = describe(index, queries)

    assert "  passages:            79 (70 text, 6 tables, 3 figures)" in lines
    assert "  asked:               4 (3 answered, 1 refused, 0 failed)" in lines
    assert "  median total time:   329 s" in lines
    assert "  median search time:  81 s" in lines
    assert "  of which database:   6 ms" in lines
    assert "  median wait for the first word: no data yet" in lines
    assert "  median writing speed: 6.0 tokens per second" in lines


def test_command_line_prints_the_figures(tmp_cfg, capsys):
    register(Database(tmp_cfg.paths.sqlite_path), "a", "Document A", 10, 3)

    assert main(["--config", str(tmp_cfg.config_file)]) == 0

    output = capsys.readouterr().out
    assert "documents:           1" in output and "Questions" in output


def test_the_figures_behind_the_time_limit_are_reported_too(tmp_cfg):
    """Ranking time, the answer model's reading speed, and answers stopped at the limit."""
    db = Database(tmp_cfg.paths.sqlite_path)
    for chars, waited, stopped in ((3000, 60.0, False), (3600, 60.0, True), (2000, 25.0, False)):
        answer = logged_answer(100, rerank=18.0, first_token=waited, generation=waited + 20, tokens=100)
        timings = replace(answer.timings, evidence_chars=chars, time_limit_s=120.0, stopped_at_limit=stopped)
        db.log_answer(replace(answer, timings=timings), "e", "r")

    stats = db.query_stats()
    lines = describe(IndexStats(), stats)

    assert stats.median_rerank_s == 18.0
    assert stats.median_read_chars_per_s == pytest.approx(60.0)  # 50, 60 and 80 characters a second
    assert stats.stopped_at_limit == 1
    assert "  of which ranking:    18 s" in lines
    assert "  median reading speed: 60 characters per second" in lines
    assert "  stopped at the time limit: 1" in lines
    assert "  median reading speed: no data yet" in describe(IndexStats(), QueryStats())
