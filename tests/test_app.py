"""The chat screen, run by Streamlit's own test harness (no browser, no server).

The search models and the answer model are fakes. The index is a real Qdrant store in
a temporary folder, filled before the page is started.
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from docqa.config import CONFIG_ENV_VAR, Config
from docqa.db import Database
from docqa.index import index_chunks
from docqa.index.store import ChunkStore
from docqa.schema import ElementType
from tests.fakes import FakeEmbedder, FakeReranker, ScriptedLLM, make_chunk

NOT_FOUND = "I could not find this in the documents."
REPLY = "The processing fee on a home loan is 0.50% of the loan amount [Schedule of Charges, p. 12]."

CORPUS = [
    make_chunk(
        "fees:00000",
        "Table: Schedule of charges\nColumns: Loan type | Processing fee\nHome loan | 0.50% of the loan amount",
        doc_title="Schedule of Charges",
        page=12,
        element_type=ElementType.TABLE,
        html="<table><tr><th>Loan type</th><th>Processing fee</th></tr>"
        "<tr><td>Home loan</td><td>0.50% of the loan amount</td></tr></table>",
    ),
    make_chunk("hours:00000", "Branches open at ten and close at five.", doc_title="Branch Manual", page=2),
]


def run_page(project_root) -> AppTest:
    page = AppTest.from_file(str(project_root / "src" / "docqa" / "app" / "main.py"))
    return page.run(timeout=120)


def fill_index(config: Config) -> None:
    """Index the corpus, then close the store so that the page can open it."""
    store = ChunkStore(config.qdrant)
    try:
        index_chunks(CORPUS, config, embedder=FakeEmbedder(), store=store)
    finally:
        store.close()


@pytest.fixture
def ready(tmp_cfg, monkeypatch):
    """A temporary project with an index, fake search models and a scripted answer model."""
    fill_index(tmp_cfg)
    llm = ScriptedLLM(tmp_cfg, REPLY)
    monkeypatch.setenv(CONFIG_ENV_VAR, str(tmp_cfg.config_file))
    monkeypatch.setattr("docqa.resources.get_embedder", lambda config: FakeEmbedder())
    monkeypatch.setattr("docqa.resources.get_reranker", lambda config: FakeReranker())
    monkeypatch.setattr("docqa.resources.get_llm", lambda config: llm.client)
    return tmp_cfg, llm


def all_text(page: AppTest) -> str:
    return "\n".join(element.value for element in [*page.markdown, *page.caption, *page.text])


# -- before anything can be asked -----------------------------------------------------


def test_page_shows_a_configuration_problem(project_root, tmp_path, monkeypatch):
    monkeypatch.setenv(CONFIG_ENV_VAR, str(tmp_path / "missing.yaml"))

    page = run_page(project_root)

    assert not page.exception
    assert len(page.error) == 1
    assert "configuration could not be loaded" in page.error[0].value
    assert len(page.info) == 0


def test_page_says_when_no_documents_are_indexed(project_root, tmp_cfg, monkeypatch):
    monkeypatch.setenv(CONFIG_ENV_VAR, str(tmp_cfg.config_file))

    page = run_page(project_root)

    assert not page.exception
    assert page.title[0].value == "BFSI document Q&A"
    assert len(page.info) == 1
    assert "No documents are indexed yet." in page.info[0].value
    assert "python -m docqa.ingest" in page.info[0].value
    assert "qwen2.5:7b" in page.caption[0].value
    assert len(page.chat_input) == 0


def test_page_asks_for_the_threshold_before_it_takes_questions(project_root, temp_config, monkeypatch):
    def unset(raw):
        raw["retrieval"]["rerank_score_threshold"] = None

    config = temp_config(unset)
    fill_index(config)
    monkeypatch.setenv(CONFIG_ENV_VAR, str(config.config_file))

    page = run_page(project_root)

    assert not page.exception
    assert len(page.warning) == 1
    assert "rerank_score_threshold" in page.warning[0].value
    assert "--suggest-threshold" in page.warning[0].value
    assert len(page.chat_input) == 0


def test_page_explains_when_another_program_has_the_index_open(project_root, tmp_cfg, monkeypatch):
    monkeypatch.setenv(CONFIG_ENV_VAR, str(tmp_cfg.config_file))
    busy = ChunkStore(tmp_cfg.qdrant)
    try:
        page = run_page(project_root)
    finally:
        busy.close()

    assert not page.exception
    assert "in use by another program" in page.error[0].value


# -- asking ---------------------------------------------------------------------------


def test_a_question_gets_a_cited_answer_and_expandable_sources(project_root, ready):
    config, llm = ready
    page = run_page(project_root)
    assert len(page.chat_input) == 1

    page.chat_input[0].set_value("What is the processing fee on a home loan?").run(timeout=120)

    assert not page.exception
    assert [message.name for message in page.chat_message] == ["user", "assistant"]
    text = all_text(page)
    assert "What is the processing fee on a home loan?" in text
    assert REPLY in text
    assert "**Cited:** [Schedule of Charges, p. 12]" in text
    assert "by `qwen2.5:7b`" in text

    labels = [expander.label for expander in page.expander]
    assert any(label.startswith("1. Schedule of Charges, p. 12 (table, score") for label in labels)
    assert any(label.endswith("- cited") for label in labels)
    assert len(llm.requests) == 1
    assert llm.requests[0]["stream"] is True

    logged = Database(config.paths.sqlite_path).recent_queries()
    assert len(logged) == 1
    assert logged[0].question == "What is the processing fee on a home loan?"
    assert logged[0].answer == REPLY
    assert logged[0].retrieved[0]["chunk_id"] == "fees:00000"


def test_the_conversation_stays_on_screen_after_the_next_question(project_root, ready):
    page = run_page(project_root)
    page.chat_input[0].set_value("What is the processing fee on a home loan?").run(timeout=120)
    page.chat_input[0].set_value("When do branches open and close?").run(timeout=120)

    assert not page.exception
    assert [message.name for message in page.chat_message] == ["user", "assistant"] * 2
    text = all_text(page)
    assert "What is the processing fee on a home loan?" in text
    assert "When do branches open and close?" in text


def test_a_question_the_documents_cannot_answer_is_refused_without_the_model(project_root, ready):
    config, llm = ready
    page = run_page(project_root)

    page.chat_input[0].set_value("Which planets have rings?").run(timeout=120)

    assert not page.exception
    text = all_text(page)
    assert NOT_FOUND in text
    assert "The answer model was not called." in text
    assert llm.requests == []
    logged = Database(config.paths.sqlite_path).recent_queries()[0]
    assert logged.refused is True and logged.answer_model is None


def test_a_failing_model_server_is_shown_and_logged(project_root, ready, monkeypatch):
    config, _llm = ready
    broken = ScriptedLLM(config, status=500)
    monkeypatch.setattr("docqa.resources.get_llm", lambda config: broken.client)
    page = run_page(project_root)

    page.chat_input[0].set_value("What is the processing fee on a home loan?").run(timeout=120)

    assert not page.exception
    assert any("LLMResponseError" in error.value for error in page.error)
    logged = Database(config.paths.sqlite_path).recent_queries()[0]
    assert logged.error.startswith("LLMResponseError")
    assert logged.answer == ""
