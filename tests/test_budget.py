"""The time limit for an answer (generation.time_limit_s).

How much evidence fits in the time, when the writing stops, what is kept of an answer
that was stopped, and what the reader is told. The answer model is a faked server and
the clock is a counter, so nothing here waits for real time to pass.
"""

from __future__ import annotations

import pytest

from docqa.db import Database
from docqa.generate import answer, answer_body, choose_evidence, log_query, measured_read_speed
from docqa.generate.budget import AnswerPlan, complete_part, plan_answer
from docqa.generate.prompt import CUT_MARKER
from docqa.schema import RetrievalResult, RetrievalTimings, ScoredChunk
from tests.fakes import ScriptedLLM, make_chunk

TITLE = "Example Bank Lending Policy"
LONG = make_chunk(
    "fees:00001",
    "\n".join(f"Clause {number}: the processing fee on product {number} is {number} percent." for number in range(1, 61)),
    doc_id="fees",
    doc_title=TITLE,
    page=12,
)
SECOND = make_chunk(
    "fees:00002",
    "\n".join(f"Rule {number}: the prepayment charge on product {number} is nil." for number in range(1, 61)),
    doc_id="fees",
    doc_title=TITLE,
    page=13,
)


def found(*chunks, elapsed: float = 20.0) -> RetrievalResult:
    sources = tuple(ScoredChunk(chunk, 0.03, 0.9 - 0.05 * position) for position, chunk in enumerate(chunks or (LONG, SECOND)))
    return RetrievalResult(
        "q", sources, True, sources[0].rerank_score, elapsed_s=elapsed, timings=RetrievalTimings(rerank_s=elapsed, candidates=15)
    )


class Clock:
    """Stands in for the ``time`` module: every look at it is one second later."""

    def __init__(self) -> None:
        self.now = 0.0

    def perf_counter(self) -> float:
        self.now += 1.0
        return self.now


# -- how much evidence fits ---------------------------------------------------------------


def test_without_a_limit_the_whole_evidence_budget_is_used(temp_config):
    def unlimited(raw):
        raw["generation"]["time_limit_s"] = None

    config = temp_config(unlimited)

    plan = plan_answer(config, used_s=400.0)

    assert plan.evidence_chars == config.generation.evidence_char_budget
    assert (plan.time_limit_s, plan.ends_at, plan.shortened) == (None, None, False)
    assert plan.evidence_budget is None  # the evidence is cut as it always was
    assert plan.stop_writing_at(first_word_at=99.0) is None


def test_the_evidence_is_cut_to_what_can_be_read_in_the_time_left(tmp_cfg):
    """120 s, less 20 s of search, less 35 s kept for writing, leaves 65 s to read in."""
    generation = tmp_cfg.generation
    assert (generation.time_limit_s, generation.writing_reserve_s) == (120, 35)

    plan = plan_answer(tmp_cfg, used_s=20.0, read_chars_per_s=60.0, now=1000.0)

    assert plan.evidence_chars == plan.evidence_budget == 65 * 60
    assert plan.shortened is True and plan.speed_measured is True
    assert plan.ends_at == 1100.0  # the 100 s that are left of the limit
    assert plan.expected_read_s(3900) == pytest.approx(65.0)


def test_until_a_speed_is_measured_the_assumed_one_is_used(tmp_cfg):
    plan = plan_answer(tmp_cfg, used_s=20.0, now=0.0)

    assert plan.speed_measured is False
    assert plan.read_chars_per_s == tmp_cfg.generation.read_chars_per_s
    assert plan.evidence_chars == int(65 * tmp_cfg.generation.read_chars_per_s)


def test_a_fast_model_gets_the_full_budget_and_a_slow_search_still_leaves_one_passage(tmp_cfg):
    generation = tmp_cfg.generation

    fast = plan_answer(tmp_cfg, used_s=5.0, read_chars_per_s=5000.0)
    late = plan_answer(tmp_cfg, used_s=118.0, read_chars_per_s=60.0)

    assert fast.evidence_chars == generation.evidence_char_budget and fast.shortened is False
    assert late.evidence_chars == generation.min_evidence_chars and late.shortened is True


def test_writing_stops_when_the_limit_runs_out_but_never_right_after_it_started(tmp_cfg):
    plan = plan_answer(tmp_cfg, used_s=20.0, read_chars_per_s=60.0, now=1000.0)

    # The first word came in good time: writing may go on until the limit (1100).
    assert plan.stop_writing_at(first_word_at=1050.0) == 1100.0
    # The reading took longer than planned: the answer still gets its 35 s.
    assert plan.stop_writing_at(first_word_at=1090.0) == 1125.0


def test_choose_evidence_takes_the_planned_amount(tmp_cfg):
    sources = found().chunks

    everything, whole = choose_evidence(sources, tmp_cfg)
    planned, part = choose_evidence(sources, tmp_cfg, 2000)

    assert len(everything) == 2 and whole.cut is False
    assert len(planned) == 1 and part.cut is True and part.partial is True
    assert (part.included, part.offered) == (1, 2)
    # The planned amount is everything the model reads: the passage and its header.
    assert len(part.text) <= 2000 + len(CUT_MARKER) + 1 < len(LONG.text)
    assert part.text.startswith("EVIDENCE E1\n") and part.text.endswith(CUT_MARKER)


def test_the_headers_of_the_passages_count_towards_a_planned_amount(tmp_cfg):
    """Four short passages have as much header as text, and the model reads both."""
    short = [
        ScoredChunk(make_chunk(f"doc:{number:05d}", f"Clause {number} sets the fee at {number} percent. " * 4, doc_title=TITLE, page=number), 0.03, 0.9)
        for number in range(1, 9)
    ]
    text_only = sum(len(source.chunk.text) for source in short)

    by_text, evidence = choose_evidence(short, tmp_cfg)  # generation.evidence_char_budget: passage text
    by_plan, planned = choose_evidence(short, tmp_cfg, text_only)  # a planned amount: everything

    assert len(by_text) == 8 and len(evidence.text) > text_only * 1.3
    assert len(by_plan) < 8 and planned.cut is True
    assert len(planned.text) <= text_only + len(CUT_MARKER) + 1


# -- what is kept of an answer that was stopped --------------------------------------------


@pytest.mark.parametrize(
    ("written", "kept"),
    [
        ("The fee is 0.50% [E1, p. 12]. It is collected when the", "The fee is 0.50% [E1, p. 12]."),
        ("The fee is 0.50% [E1, p. 12]. Prepayment is free [E2, p. 13] and", "The fee is 0.50% [E1, p. 12]. Prepayment is free [E2, p. 13]"),
        ("- Home loan: 0.50% [E1, p. 12]\n- Car loan: 1% [E1, p.", "- Home loan: 0.50% [E1, p. 12]"),
        ("The fee is 0.50% [E1, p. 12].", "The fee is 0.50% [E1, p. 12]."),
        ("The fee is small. It is collected when the", "The fee is small."),
        ("The fee is 1 percent [E1, p.", "The fee is 1 percent"),
        ("The processing fee on", "The processing fee on"),
        ("", ""),
    ],
)
def test_a_stopped_answer_is_kept_up_to_its_last_complete_statement(written, kept):
    assert complete_part(written) == kept


# -- the answer itself ---------------------------------------------------------------------


def test_an_answer_within_its_time_is_not_touched(tmp_cfg):
    llm = ScriptedLLM(tmp_cfg, "The fee on product 1 is 1 percent [E1, p. 12].")

    result = answer("fee?", found(LONG, elapsed=1.0), llm.client, tmp_cfg)

    assert result.text == f"The fee on product 1 is 1 percent [{TITLE}, p. 12]."
    assert result.notes == ()
    assert (result.timings.time_limit_s, result.timings.stopped_at_limit) == (120, False)
    assert result.timings.completion_tokens == 20


def test_an_answer_is_stopped_at_the_limit_and_ends_with_a_complete_statement(tmp_cfg, monkeypatch):
    reply = "The fee is 1 percent [E1, p. 12]. It is collected when the loan is sanctioned and then again every year"
    llm = ScriptedLLM(tmp_cfg, reply)
    monkeypatch.setattr("docqa.generate.time", Clock())
    # The clock reads 3 when the first word arrives. The limit runs out at 11, which is
    # when the ninth word has come: "The fee is 1 percent [E1, p. 12]. It".
    plan = AnswerPlan(evidence_chars=8000, time_limit_s=120, ends_at=11.0, writing_reserve_s=2.0)
    pieces: list[str] = []

    result = answer("fee?", found(LONG), llm.client, tmp_cfg, on_text=pieces.append, plan=plan)

    assert "".join(pieces).startswith("The fee is 1 percent [E1, p. 12]. It ")
    assert "sanctioned" not in "".join(pieces)  # the model was not left writing
    assert answer_body(result) == f"The fee is 1 percent [{TITLE}, p. 12]."
    assert result.refused is False and [c.page for c in result.citations] == [12]
    assert any("stopped because the time limit" in note and "120 s" in note for note in result.notes)
    assert result.timings.stopped_at_limit is True
    assert result.timings.completion_tokens is None  # the server sends the count only at the end


def test_an_answer_whose_reading_ran_late_still_gets_its_time_to_be_written(tmp_cfg, monkeypatch):
    """The limit ran out before the first word. Writing is given writing_reserve_s all the same."""
    llm = ScriptedLLM(tmp_cfg, "The fee is 1 percent [E1, p. 12]. It is collected at sanction [E1, p. 12]. And more words follow here")
    monkeypatch.setattr("docqa.generate.time", Clock())
    plan = AnswerPlan(evidence_chars=8000, time_limit_s=120, ends_at=1.0, writing_reserve_s=7.0)

    result = answer("fee?", found(LONG), llm.client, tmp_cfg, plan=plan)

    # First word at 3, so writing may go on until 10: eight words, "... [E1, p. 12]."
    assert answer_body(result) == f"The fee is 1 percent [{TITLE}, p. 12]."
    assert result.timings.stopped_at_limit is True


def test_a_refusal_written_by_the_model_is_not_stopped_into_something_else(tmp_cfg):
    llm = ScriptedLLM(tmp_cfg, "I could not find this in the documents.")

    result = answer("weather?", found(LONG), llm.client, tmp_cfg)

    assert result.refused is True and result.timings.stopped_at_limit is False


def test_the_reader_is_told_when_the_limit_left_out_part_of_what_was_found(tmp_cfg):
    llm = ScriptedLLM(tmp_cfg, "The fee on product 1 is 1 percent [E1, p. 12].")
    plan = plan_answer(tmp_cfg, used_s=20.0, read_chars_per_s=40.0)  # room for 2,600 characters
    assert plan.shortened is True

    result = answer("fee?", found(), llm.client, tmp_cfg, plan=plan)

    prompt = llm.requests[0]["messages"][1]["content"]
    assert "EVIDENCE E1" in prompt and "EVIDENCE E2" not in prompt
    assert result.evidence_ids == ("fees:00001",)
    note = next(note for note in result.notes if "time limit" in note)
    assert "120 s" in note and "only part of the best passage" in note
    assert result.timings.evidence_chars <= plan.evidence_chars + len(CUT_MARKER) + 1


def test_the_note_counts_the_passages_when_some_were_read_whole(tmp_cfg):
    llm = ScriptedLLM(tmp_cfg, "The fee on product 1 is 1 percent [E1, p. 12].")
    whole_first = len(LONG.text) + 800  # room for the first passage, both headers and a little of the second
    plan = AnswerPlan(evidence_chars=whole_first, time_limit_s=120, ends_at=None, shortened=True)

    result = answer("fee?", found(), llm.client, tmp_cfg, plan=plan)

    note = next(note for note in result.notes if "time limit" in note)
    assert "2 of the 2 passages worth reading, the last of them cut short" in note


def test_without_a_plan_the_answer_plans_from_the_time_the_search_took(tmp_cfg):
    llm = ScriptedLLM(tmp_cfg, "The fee on product 1 is 1 percent [E1, p. 12].")

    quick = answer("fee?", found(elapsed=1.0), llm.client, tmp_cfg)
    slow = answer("fee?", found(elapsed=110.0), llm.client, tmp_cfg)

    assert quick.timings.evidence_chars > slow.timings.evidence_chars
    assert slow.timings.evidence_chars <= tmp_cfg.generation.min_evidence_chars + len(CUT_MARKER) + 1


def test_the_steps_of_an_answer_are_reported(tmp_cfg):
    llm = ScriptedLLM(tmp_cfg, "The fee on product 1 is 1 percent [E1, p. 12].")
    seen: list[tuple[str, str, dict]] = []

    answer("fee?", found(LONG), llm.client, tmp_cfg, on_step=lambda step, state, details: seen.append((step, state, details)))

    assert [(step, state) for step, state, _ in seen] == [
        ("reading", "start"),
        ("reading", "end"),
        ("writing", "start"),
        ("writing", "end"),
    ]
    start = seen[0][2]
    assert start["passages"] == 1 and start["chars"] > 1000 and start["expected_s"] > 0
    assert seen[1][2]["took_s"] >= 0 and seen[3][2]["stopped"] is False


def test_nothing_is_reported_and_no_model_is_asked_when_nothing_was_found(tmp_cfg):
    llm = ScriptedLLM(tmp_cfg, "never asked")
    seen: list[str] = []
    weak = RetrievalResult("q", found().chunks, found=False, best_score=0.1)

    result = answer("weather?", weak, llm.client, tmp_cfg, on_step=lambda step, state, details: seen.append(step))

    assert result.refused is True and seen == [] and llm.requests == []


# -- the reading speed, measured ------------------------------------------------------------


def test_the_reading_speed_comes_from_the_answers_already_given(tmp_cfg):
    assert measured_read_speed(tmp_cfg) is None  # nothing has been answered yet
    llm = ScriptedLLM(tmp_cfg, "The fee on product 1 is 1 percent [E1, p. 12].")

    for _ in range(3):
        log_query(answer("fee?", found(LONG, elapsed=1.0), llm.client, tmp_cfg), tmp_cfg)

    speed = measured_read_speed(tmp_cfg)
    logged = Database(tmp_cfg.paths.sqlite_path).recent_queries()[0].timings
    assert speed is not None and speed > 0
    assert speed == pytest.approx(logged["evidence_chars"] / logged["first_token_s"], rel=3.0)
    assert plan_answer(tmp_cfg, used_s=20.0, read_chars_per_s=speed).speed_measured is True


def test_the_reading_speed_is_the_median_for_the_answer_model_in_use(tmp_path):
    database = Database(tmp_path / "docqa.sqlite")

    def log(model: str, chars: int, waited: float, **more) -> None:
        database._log(
            question="q",
            answer="a",
            refused=more.pop("refused", False),
            error=more.pop("error", None),
            retrieved=[],
            citations=[],
            latency_s=waited + 5,
            answer_model=model,
            embedding_model="e",
            reranker_model="r",
            timings={"first_token_s": waited, "evidence_chars": chars, **more},
        )

    for chars, waited in ((3000, 60.0), (3000, 50.0), (3000, 30.0)):  # 50, 60 and 100 characters a second
        log("qwen2.5:7b", chars, waited)
    log("another-model:1b", 3000, 3.0)  # a faster model does not count for this one
    log("qwen2.5:7b", 3000, 1.0, refused=True)  # nor does a refusal
    log("qwen2.5:7b", 3000, 1.0, error="LLMTimeoutError")  # nor a failure
    log("qwen2.5:7b", 0, 10.0)  # nor an answer without evidence

    assert database.read_speed("qwen2.5:7b") == pytest.approx(60.0)
    assert database.read_speed("QWEN2.5:7B") == pytest.approx(60.0)
    assert database.read_speed("another-model:1b") == pytest.approx(1000.0)
    assert database.read_speed("another-model") is None  # "name" means "name:latest"
    assert database.read_speed("not-used:7b") is None
    assert database.read_speed("qwen2.5:7b", sample=1) == pytest.approx(100.0)  # the latest answer only


def test_a_row_logged_by_the_release_before_this_one_gives_a_speed_too(tmp_path):
    """Release 2026-11-23 logged the evidence size and the wait for the first word already."""
    database = Database(tmp_path / "docqa.sqlite")
    database._log(
        question="q", answer="a", refused=False, error=None, retrieved=[], citations=[], latency_s=70.0,
        answer_model="qwen2.5:7b", embedding_model="e", reranker_model="r",
        timings={"first_token_s": 50.0, "evidence_chars": 4000},
    )

    assert database.read_speed("qwen2.5:7b") == pytest.approx(80.0)
    stats = database.query_stats()
    assert stats.median_read_chars_per_s == pytest.approx(80.0) and stats.stopped_at_limit == 0
