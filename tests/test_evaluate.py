"""The evaluation script: reading questions, running them, measuring, reporting.

Search models and the answer model are fakes; the index is a real Qdrant store in a
temporary folder. The numbers these tests produce say nothing about real quality.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from docqa.config import ConfigError
from docqa.db import Database, DocumentRecord, utc_now
from docqa.evaluate import (
    EvaluationError,
    Question,
    QuestionResult,
    doc_ids_for,
    evaluate,
    load_questions,
    main,
    suggest_threshold,
    summarise,
    write_results,
)
from docqa.index import index_chunks
from docqa.index.store import ChunkStore
from docqa.schema import ElementType
from tests.fakes import FakeEmbedder, FakeReranker, ScriptedLLM, make_chunk

HEADER = "question,expected_answer,source_doc,page,type\n"
CORPUS = [
    make_chunk(
        "fees:00000",
        "Table: Schedule of charges\nColumns: Loan type | Processing fee\nHome loan | 0.50% of the loan amount",
        doc_title="Schedule of Charges",
        page=12,
        element_type=ElementType.TABLE,
    ),
    make_chunk(
        "manual:00000",
        "Branches open at ten in the morning. Branches close at five in the evening.",
        doc_title="Branch Manual",
        page=2,
        page_end=3,
        page_marks=((37, 3),),
    ),
]
QUESTIONS = (
    HEADER
    + '"What is the processing fee on a home loan?",0.50% of the loan amount,Schedule of Charges.pdf,12,table\n'
    + "When do branches close in the evening?,At five in the evening,manuals/Branch Manual.pdf,3,text\n"
    + "Which planets have rings?,,,,not_in_docs\n"
)


def scripted_reply(body: dict) -> str:
    question = body["messages"][1]["content"].split("QUESTION\n")[-1]
    if "processing fee" in question:
        return "It is 0.50% of the loan amount [Schedule of Charges, p. 12]."
    return "Branches close at five in the evening [Branch Manual, p. 2]."


def register(db: Database, doc_id: str, source_path: str, title: str) -> None:
    db.save_document(
        DocumentRecord(doc_id, source_path, "a" * 64, "f" * 64, title, None, None, "all", 20, 1, utc_now())
    )


@pytest.fixture
def setup(tmp_cfg, store):
    """An index with two documents, both in the register, and the fakes to run against it."""
    index_chunks(CORPUS, tmp_cfg, embedder=FakeEmbedder(), store=store)
    db = Database(tmp_cfg.paths.sqlite_path)
    register(db, "fees", "Schedule of Charges.pdf", "Schedule of Charges")
    register(db, "manual", "manuals/Branch Manual.pdf", "Branch Manual")
    llm = ScriptedLLM(tmp_cfg, scripted_reply)
    return {
        "config": tmp_cfg,
        "llm": llm,
        "options": {
            "llm": llm.client,
            "embedder": FakeEmbedder(),
            "store": store,
            "reranker": FakeReranker(),
            "db": db,
        },
    }


def questions_file(folder: Path, text: str = QUESTIONS, encoding: str = "utf-8") -> Path:
    path = folder / "questions.csv"
    path.write_text(text, encoding=encoding)
    return path


# -- the question file ----------------------------------------------------------------


def test_questions_are_read_with_their_row_numbers(tmp_path):
    questions = load_questions(questions_file(tmp_path, QUESTIONS + ",,,,\n", encoding="utf-8-sig"))

    assert [q.number for q in questions] == [2, 3, 4]
    first, _second, third = questions
    assert first == Question(
        2, "What is the processing fee on a home loan?", "0.50% of the loan amount", "Schedule of Charges.pdf", 12, "table"
    )
    assert first.answerable and not third.answerable
    assert (third.source_doc, third.page) == (None, None)


def test_the_projects_own_question_file_has_the_right_columns_and_no_rows_yet(config):
    assert load_questions(config.paths.eval_questions) == []


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        pytest.param("question,expected_answer,source_doc,type\nQ?,A,a.pdf,text\n", "missing column", id="no page column"),
        pytest.param(HEADER + ",A,a.pdf,3,text\n", "'question' is empty", id="empty question"),
        pytest.param(HEADER + "Q?,A,a.pdf,3,\n", "'type' is empty", id="empty type"),
        pytest.param(HEADER + "Q?,A,a.pdf,three,text\n", "page number", id="page not a number"),
        pytest.param(HEADER + "Q?,A,a.pdf,0,text\n", "page number", id="page zero"),
        pytest.param(HEADER + "Q?,A,,,text\n", "'source_doc' and 'page' are needed", id="no source"),
    ],
)
def test_question_file_mistakes_are_named_with_their_row(tmp_path, content, expected):
    with pytest.raises(EvaluationError, match=expected):
        load_questions(questions_file(tmp_path, content))
    with pytest.raises(EvaluationError, match="does not exist"):
        load_questions(tmp_path / "nowhere.csv")


def test_source_doc_is_matched_by_file_name_or_by_path(setup):
    db = setup["options"]["db"]

    assert doc_ids_for("Schedule of Charges.pdf", db) == {"fees"}
    assert doc_ids_for("branch manual.PDF", db) == {"manual"}
    assert doc_ids_for("manuals\\Branch Manual.pdf", db) == {"manual"}
    assert doc_ids_for("Annual Report.pdf", db) == set()


# -- running --------------------------------------------------------------------------


def test_each_question_is_measured(setup, tmp_path):
    questions = load_questions(questions_file(tmp_path))

    table, text, outside = evaluate(questions, setup["config"], **setup["options"])

    assert (table.refused, table.refusal_correct) == (False, True)
    assert table.page_in_top_n is True
    assert table.cites_expected_page is True
    assert table.citations_valid is True
    assert table.citations == ("[Schedule of Charges, p. 12]",)
    assert table.answer.startswith("It is 0.50%")
    assert table.top_passages[0].startswith("Schedule of Charges p. 12 (")
    assert table.error is None and table.latency_s > 0

    # The search found the right pages (the chunk covers pages 2 to 3), but the answer
    # cites page 2 while the fact is on page 3.
    assert text.page_in_top_n is True
    assert text.cites_expected_page is False
    assert text.citations_valid is True

    assert (outside.refused, outside.refusal_correct) == (True, True)
    assert outside.answer == "I could not find this in the documents."
    assert (outside.page_in_top_n, outside.cites_expected_page, outside.citations_valid) == (None, None, None)
    assert len(setup["llm"].requests) == 2  # the refused question never reached the model


def test_search_can_be_judged_without_an_answer_model_or_a_threshold(temp_config, tmp_path):
    def unset(raw):
        raw["retrieval"]["rerank_score_threshold"] = None

    config = temp_config(unset)
    store = ChunkStore(config.qdrant)
    try:
        index_chunks(CORPUS, config, embedder=FakeEmbedder(), store=store)
        db = Database(config.paths.sqlite_path)
        register(db, "fees", "Schedule of Charges.pdf", "Schedule of Charges")
        questions = load_questions(questions_file(tmp_path))
        options = {"embedder": FakeEmbedder(), "store": store, "reranker": FakeReranker(), "db": db}

        results = evaluate(questions, config, retrieval_only=True, **options)
        with pytest.raises(ConfigError, match="--suggest-threshold"):
            evaluate(questions, config, **options)
    finally:
        store.close()

    table, text, outside = results
    assert table.page_in_top_n is True
    assert 0 < table.best_score <= 1
    assert (table.refused, table.answer, table.citations_valid) == (None, "", None)
    # Branch Manual was never registered here, so the question's document is unknown.
    assert text.page_in_top_n is False
    assert "is not among the ingested documents" in text.error
    assert outside.page_in_top_n is None and outside.best_score is not None
    assert outside.best_score < table.best_score


def test_a_question_that_fails_is_recorded_and_the_run_goes_on(setup, tmp_path):
    broken = ScriptedLLM(setup["config"], status=500)
    options = {**setup["options"], "llm": broken.client}

    results = evaluate(load_questions(questions_file(tmp_path)), setup["config"], **options)

    assert results[0].error.startswith("LLMResponseError")
    assert results[0].refused is None
    assert results[2].refused is True and results[2].error is None


# -- the threshold --------------------------------------------------------------------


def result_with_score(kind: str, score: float | None) -> QuestionResult:
    return QuestionResult(Question(2, "q", "", "a.pdf" if kind != "not_in_docs" else None, 1, kind), best_score=score)


def test_the_threshold_goes_in_the_middle_of_the_gap_between_the_two_groups():
    results = [result_with_score("text", s) for s in (0.92, 0.81, 0.74)] + [
        result_with_score("not_in_docs", s) for s in (0.10, 0.22)
    ]

    suggestion = suggest_threshold(results)

    assert suggestion.threshold == pytest.approx(0.48)
    assert (suggestion.answerable_kept, suggestion.answerable_total) == (3, 3)
    assert (suggestion.unanswerable_refused, suggestion.unanswerable_total) == (2, 2)


def test_overlapping_groups_get_the_threshold_with_the_best_balance():
    results = [result_with_score("table", s) for s in (0.9, 0.8, 0.7, 0.2)] + [
        result_with_score("not_in_docs", s) for s in (0.1, 0.3, 0.35)
    ]

    suggestion = suggest_threshold(results)

    assert suggestion.threshold == pytest.approx(0.525)
    assert (suggestion.answerable_kept, suggestion.unanswerable_refused) == (3, 3)


def test_no_threshold_is_suggested_from_one_kind_of_question_alone():
    assert suggest_threshold([result_with_score("text", 0.9)]) is None
    assert suggest_threshold([result_with_score("not_in_docs", 0.1)]) is None
    assert suggest_threshold([result_with_score("text", None), result_with_score("not_in_docs", 0.1)]) is None


# -- reporting ------------------------------------------------------------------------


def test_results_are_summarised_per_type_and_written_with_columns_for_manual_judgement(setup, tmp_path):
    results = evaluate(load_questions(questions_file(tmp_path)), setup["config"], **setup["options"])

    rows = summarise(results)
    folder = write_results(results, setup["config"], tmp_path / "run", retrieval_only=False)

    by_type = {row["type"]: row for row in rows}
    assert list(by_type) == ["table", "text", "not_in_docs", "all"]
    assert by_type["table"]["page in top 5"] == "1/1"
    assert by_type["text"]["cites expected page"] == "0/1"
    assert by_type["not_in_docs"]["page in top 5"] == "-"
    assert by_type["all"]["refusal correct"] == "3/3"
    assert by_type["all"]["questions"] == "3"

    with open(folder / "results.csv", encoding="utf-8-sig", newline="") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == 3
    assert written[0]["page_in_top_5"] == "yes" and written[2]["refused"] == "yes"
    assert written[0]["citations"] == "[Schedule of Charges, p. 12]"
    assert {row["manual_correct"] for row in written} == {""}
    assert "manual_notes" in written[0]

    summary = (folder / "summary.md").read_text(encoding="utf-8")
    assert "| type | questions | page in top 5 |" in summary
    assert "Answer model: qwen2.5:7b" in summary
    assert "manual_correct" in summary


# -- command line ---------------------------------------------------------------------


def test_command_line_runs_writes_a_folder_and_suggests_a_threshold(setup, monkeypatch, capsys):
    config = setup["config"]
    config.paths.eval_questions.parent.mkdir(parents=True, exist_ok=True)
    assert questions_file(config.paths.eval_questions.parent) == config.paths.eval_questions
    options = setup["options"]
    monkeypatch.setattr("docqa.resources.get_embedder", lambda config: options["embedder"])
    monkeypatch.setattr("docqa.resources.get_reranker", lambda config: options["reranker"])
    monkeypatch.setattr("docqa.resources.get_store", lambda config: options["store"])
    monkeypatch.setattr("docqa.resources.get_llm", lambda config: options["llm"])
    config_file = str(config.config_file)

    assert main(["--config", config_file]) == 0
    output = capsys.readouterr().out
    assert "type: all" in output and "Results written to" in output
    runs = list(config.paths.eval_results_dir.iterdir())
    assert len(runs) == 1 and (runs[0] / "results.csv").is_file()

    out = config.paths.eval_results_dir / "threshold-run"
    assert main(["--config", config_file, "--suggest-threshold", "--out", str(out)]) == 0
    output = capsys.readouterr().out
    assert "retrieval.rerank_score_threshold:" in output
    assert "## Suggested threshold" in (out / "summary.md").read_text(encoding="utf-8")

    assert main(["--config", config_file, "--type", "table", "--limit", "1", "--retrieval-only", "--out", str(out)]) == 0
    assert "questions: 1" in capsys.readouterr().out


def test_command_line_says_when_there_is_nothing_to_run(tmp_cfg, capsys):
    tmp_cfg.paths.eval_questions.parent.mkdir(parents=True, exist_ok=True)
    tmp_cfg.paths.eval_questions.write_text(HEADER, encoding="utf-8")

    assert main(["--config", str(tmp_cfg.config_file)]) == 1
    assert "There are no questions to run" in capsys.readouterr().out


def test_the_questions_can_be_run_with_another_answer_model(setup, monkeypatch):
    """--answer-model: two models are compared on the same questions without editing config.yaml."""
    from docqa.config import load_config
    from docqa.evaluate import with_answer_model

    config = setup["config"]
    config.paths.eval_questions.parent.mkdir(parents=True, exist_ok=True)
    questions_file(config.paths.eval_questions.parent)
    options = setup["options"]
    servers: list[ScriptedLLM] = []

    def llm_for(given):
        servers.append(ScriptedLLM(given, scripted_reply))
        return servers[-1].client

    monkeypatch.setattr("docqa.resources.get_embedder", lambda config: options["embedder"])
    monkeypatch.setattr("docqa.resources.get_reranker", lambda config: options["reranker"])
    monkeypatch.setattr("docqa.resources.get_store", lambda config: options["store"])
    monkeypatch.setattr("docqa.resources.get_llm", llm_for)
    out = config.paths.eval_results_dir / "other-model"

    assert main(["--config", str(config.config_file), "--answer-model", "qwen3:4b-instruct", "--out", str(out)]) == 0

    requests = [request for server in servers for request in server.requests]
    assert requests and all(request["model"] == "qwen3:4b-instruct" for request in requests)
    assert "- Answer model: qwen3:4b-instruct" in (out / "summary.md").read_text(encoding="utf-8")
    # The file on disk is as it was, and so is everything else in the settings.
    assert load_config(config.config_file).llm.answer_model == "qwen2.5:7b"
    changed = with_answer_model(config, "another:1b")
    assert changed.llm.answer_model == "another:1b" and changed.llm.vision_model == config.llm.vision_model
    assert changed.paths == config.paths and changed.config_file == config.config_file
