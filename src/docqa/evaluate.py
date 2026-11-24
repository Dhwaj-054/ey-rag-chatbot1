"""Run the test questions and report how the chatbot did, per question type.

    python -m docqa.evaluate                       every question, search and answer
    python -m docqa.evaluate --retrieval-only      search only: no answer model needed
    python -m docqa.evaluate --suggest-threshold   propose retrieval.rerank_score_threshold

The questions are in ``paths.eval_questions`` (see eval/README.md). Each run writes a
new folder under ``paths.eval_results_dir``:

    results.csv   one row per question, with two empty columns, ``manual_correct`` and
                  ``manual_notes``, for a person who knows the documents to fill in
    summary.md    the measures below per question type, and the settings of the run

What is measured, per question type:

- **Correct page in the top 5.** Is the page that holds the answer among the five best
  passages after reranking? This judges the search on its own.
- **Cites the expected page.** Does the answer cite the document and page that hold the
  answer? Counted over the questions that got an answer.
- **Citations valid.** Does every citation in the answer point at a passage the model
  was actually given?
- **Refusal accuracy.** ``not_in_docs`` questions must be refused; all others must not.
- **Latency.** Seconds per question, search and answer together.

Whether an answer is *right* is not measured here. That is what ``manual_correct`` is
for: a wrong answer can cite the right page.

Stop the chat app first. Both use the same index, and Qdrant's local mode lets only
one program open it.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from docqa import resources
from docqa.config import Config, ConfigError, load_config
from docqa.db import Database
from docqa.generate import answer
from docqa.generate.prompt import PROMPT_VERSION
from docqa.index.embedder import Embedder, EmbeddingError
from docqa.index.store import ChunkStore, StoreError
from docqa.llm import LLMClient, LLMError
from docqa.retrieve import rank_candidates, retrieve
from docqa.retrieve.reranker import Reranker, RerankError
from docqa.schema import ScoredChunk

QUESTION_COLUMNS = ("question", "expected_answer", "source_doc", "page", "type")
NOT_IN_DOCS = "not_in_docs"
TOP_N = 5  # "correct page in top 5" (CLAUDE.md)


class EvaluationError(Exception):
    """The evaluation cannot run, for example because the question file is not usable."""


@dataclass(frozen=True)
class Question:
    """One row of the question file."""

    number: int
    """Row number in the file, counting the header as row 1."""
    question: str
    expected_answer: str
    source_doc: str | None
    page: int | None
    type: str

    @property
    def answerable(self) -> bool:
        return self.type != NOT_IN_DOCS


@dataclass(frozen=True)
class QuestionResult:
    """What happened for one question. ``None`` means "does not apply to this question"."""

    question: Question
    answer: str = ""
    refused: bool | None = None
    refusal_correct: bool | None = None
    best_score: float | None = None
    page_in_top_n: bool | None = None
    top_passages: tuple[str, ...] = ()
    cites_expected_page: bool | None = None
    citations_valid: bool | None = None
    citations: tuple[str, ...] = ()
    unmatched_citations: tuple[str, ...] = ()
    latency_s: float = 0.0
    error: str | None = None


@dataclass(frozen=True)
class ThresholdSuggestion:
    """A proposed ``retrieval.rerank_score_threshold`` and how well it separates the two groups."""

    threshold: float
    answerable_kept: int
    """Answerable questions whose best score is at or above the threshold."""
    answerable_total: int
    unanswerable_refused: int
    """``not_in_docs`` questions whose best score is below the threshold."""
    unanswerable_total: int


# -- the question file ----------------------------------------------------------------


def load_questions(path: Path) -> list[Question]:
    """Read the question file.

    Raises:
        EvaluationError: The file is missing, a column is missing, or a row cannot be
            used. The message names the row.
    """
    if not path.is_file():
        raise EvaluationError(f"The question file does not exist: {path}")
    questions: list[Question] = []
    try:
        with open(path, encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = [name.strip() for name in reader.fieldnames or []]
            missing = [name for name in QUESTION_COLUMNS if name not in columns]
            if missing:
                raise EvaluationError(f"{path}: missing column(s): {', '.join(missing)}.")
            for number, row in enumerate(reader, start=2):
                cells = {(key or "").strip(): (value or "").strip() for key, value in row.items()}
                if not any(cells.values()):
                    continue
                where = f"{path}, row {number}"
                if not cells["question"]:
                    raise EvaluationError(f"{where}: 'question' is empty.")
                kind = cells["type"].lower()
                if not kind:
                    raise EvaluationError(f"{where}: 'type' is empty.")
                page: int | None = None
                if cells["page"]:
                    if not cells["page"].isdigit() or int(cells["page"]) < 1:
                        raise EvaluationError(f"{where}: 'page' must be a page number, counted from 1.")
                    page = int(cells["page"])
                if kind != NOT_IN_DOCS and (not cells["source_doc"] or page is None):
                    raise EvaluationError(
                        f"{where}: 'source_doc' and 'page' are needed for every question whose "
                        f"type is not {NOT_IN_DOCS}."
                    )
                questions.append(
                    Question(
                        number=number,
                        question=cells["question"],
                        expected_answer=cells["expected_answer"],
                        source_doc=cells["source_doc"] or None,
                        page=page,
                        type=kind,
                    )
                )
    except (OSError, csv.Error, UnicodeDecodeError) as exc:
        raise EvaluationError(f"{path} could not be read: {exc}") from exc
    return questions


def doc_ids_for(source_doc: str, db: Database) -> set[str]:
    """The ingested documents a ``source_doc`` cell refers to.

    The cell holds a file name, or a path below ``docs/``. A bare file name matches
    every ingested file of that name.
    """
    wanted = source_doc.replace("\\", "/").strip().strip("/").lower()
    found = set()
    for record in db.list_documents():
        path = record.source_path.replace("\\", "/").lower()
        if path == wanted or path.rsplit("/", 1)[-1] == wanted or path.endswith("/" + wanted):
            found.add(record.doc_id)
    return found


# -- running --------------------------------------------------------------------------


def evaluate(
    questions: Sequence[Question],
    config: Config,
    *,
    retrieval_only: bool = False,
    llm: LLMClient | None = None,
    embedder: Embedder | None = None,
    store: ChunkStore | None = None,
    reranker: Reranker | None = None,
    db: Database | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[QuestionResult]:
    """Run every question and return one result per question, in the same order.

    With ``retrieval_only`` the answer model is not used and no threshold is needed:
    only the search is judged. A question that fails (the model server is down, say)
    gets its error recorded and the run goes on.
    """
    say = progress or (lambda line: None)
    db = db or resources.get_database(config)
    embedder = embedder or resources.get_embedder(config)
    store = store or resources.get_store(config)
    reranker = reranker or resources.get_reranker(config)
    if not retrieval_only:
        llm = llm or resources.get_llm(config)
        if config.retrieval.rerank_score_threshold is None:
            raise ConfigError(
                "retrieval.rerank_score_threshold is not set in config.yaml. Run with "
                "--suggest-threshold first, or with --retrieval-only to judge the search alone."
            )
    models = {"embedder": embedder, "store": store, "reranker": reranker}

    results: list[QuestionResult] = []
    for position, question in enumerate(questions, start=1):
        say(f"[{position}/{len(questions)}] {question.question}")
        doc_ids = doc_ids_for(question.source_doc, db) if question.source_doc else set()
        started = time.perf_counter()
        try:
            if retrieval_only:
                ranked = rank_candidates(question.question, config, **models)
                results.append(
                    QuestionResult(
                        question=question,
                        best_score=ranked[0].rerank_score if ranked else None,
                        page_in_top_n=_page_found(question, ranked, doc_ids),
                        top_passages=_passages(ranked),
                        latency_s=time.perf_counter() - started,
                        error=_not_ingested(question, doc_ids),
                    )
                )
                continue

            retrieval = retrieve(question.question, config, **models)
            reply = answer(question.question, retrieval, llm, config)
            by_id = {source.chunk.chunk_id: source.chunk for source in reply.sources}
            answered = not reply.refused
            results.append(
                QuestionResult(
                    question=question,
                    answer=reply.text,
                    refused=reply.refused,
                    refusal_correct=reply.refused == (not question.answerable),
                    best_score=retrieval.best_score,
                    page_in_top_n=_page_found(question, retrieval.chunks, doc_ids),
                    top_passages=_passages(retrieval.chunks),
                    cites_expected_page=(
                        any(
                            citation.page == question.page and by_id[citation.chunk_id].doc_id in doc_ids
                            for citation in reply.citations
                        )
                        if answered and question.answerable
                        else None
                    ),
                    citations_valid=(
                        bool(reply.citations) and not reply.unmatched_citations if answered else None
                    ),
                    citations=tuple(f"[{c.doc_title}, p. {c.page}]" for c in reply.citations),
                    unmatched_citations=reply.unmatched_citations,
                    latency_s=reply.latency_s,
                    error=_not_ingested(question, doc_ids),
                )
            )
        except (LLMError, EmbeddingError, RerankError, StoreError) as exc:
            say(f"    FAILED: {type(exc).__name__}: {exc}")
            results.append(
                QuestionResult(
                    question=question,
                    latency_s=time.perf_counter() - started,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    return results


def suggest_threshold(results: Sequence[QuestionResult]) -> ThresholdSuggestion | None:
    """Choose the threshold that best separates answerable from ``not_in_docs`` questions.

    Uses each question's best rerank score. Every midpoint between two neighbouring
    scores is tried, and the one with the highest balanced accuracy wins: the mean of
    the share of answerable questions kept and the share of ``not_in_docs`` questions
    refused. Among equally good ones, the middle of the widest gap is taken.

    Returns ``None`` when one of the two groups has no scores: the threshold cannot be
    chosen from one kind of question alone.
    """
    answerable = sorted(r.best_score for r in results if r.question.answerable and r.best_score is not None)
    unanswerable = sorted(
        r.best_score for r in results if not r.question.answerable and r.best_score is not None
    )
    if not answerable or not unanswerable:
        return None

    scores = sorted(set(answerable) | set(unanswerable))
    candidates = [(low + high) / 2 for low, high in zip(scores, scores[1:], strict=False)]
    candidates = candidates or [scores[0]]
    gaps = dict(zip(candidates, (high - low for low, high in zip(scores, scores[1:], strict=False)), strict=False))

    def quality(threshold: float) -> tuple[float, float]:
        kept = sum(score >= threshold for score in answerable) / len(answerable)
        refused = sum(score < threshold for score in unanswerable) / len(unanswerable)
        return (kept + refused) / 2, gaps.get(threshold, 0.0)

    best = max(candidates, key=quality)
    return ThresholdSuggestion(
        threshold=round(best, 4),
        answerable_kept=sum(score >= best for score in answerable),
        answerable_total=len(answerable),
        unanswerable_refused=sum(score < best for score in unanswerable),
        unanswerable_total=len(unanswerable),
    )


# -- reporting ------------------------------------------------------------------------


def summarise(results: Sequence[QuestionResult]) -> list[dict[str, str]]:
    """One row per question type, then a row for all questions together."""
    kinds = list(dict.fromkeys(result.question.type for result in results))
    rows = [_summary_row(kind, [r for r in results if r.question.type == kind]) for kind in kinds]
    if len(kinds) > 1:
        rows.append(_summary_row("all", list(results)))
    return rows


def write_results(
    results: Sequence[QuestionResult],
    config: Config,
    folder: Path,
    *,
    retrieval_only: bool,
    suggestion: ThresholdSuggestion | None = None,
) -> Path:
    """Write ``results.csv`` and ``summary.md`` into ``folder``. Returns the folder."""
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "results.csv", "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "row", "type", "question", "expected_answer", "source_doc", "page", "answer",
                "refused", "refusal_correct", "best_rerank_score", f"page_in_top_{TOP_N}",
                f"top_{TOP_N}_passages", "cites_expected_page", "citations_valid", "citations",
                "unmatched_citations", "latency_s", "error", "manual_correct", "manual_notes",
            ]
        )  # fmt: skip
        for result in results:
            question = result.question
            writer.writerow(
                [
                    question.number, question.type, question.question, question.expected_answer,
                    question.source_doc or "", question.page or "", result.answer,
                    _flag(result.refused), _flag(result.refusal_correct),
                    "" if result.best_score is None else f"{result.best_score:.4f}",
                    _flag(result.page_in_top_n), "; ".join(result.top_passages),
                    _flag(result.cites_expected_page), _flag(result.citations_valid),
                    " ".join(result.citations), " ".join(result.unmatched_citations),
                    f"{result.latency_s:.1f}", result.error or "", "", "",
                ]
            )  # fmt: skip
    (folder / "summary.md").write_text(
        _summary_markdown(results, config, retrieval_only, suggestion), encoding="utf-8"
    )
    return folder


def _summary_row(kind: str, results: list[QuestionResult]) -> dict[str, str]:
    latencies = [result.latency_s for result in results if result.error is None or result.latency_s]
    return {
        "type": kind,
        "questions": str(len(results)),
        f"page in top {TOP_N}": _rate([r.page_in_top_n for r in results]),
        "cites expected page": _rate([r.cites_expected_page for r in results]),
        "citations valid": _rate([r.citations_valid for r in results]),
        "refusal correct": _rate([r.refusal_correct for r in results]),
        "mean latency (s)": f"{statistics.mean(latencies):.1f}" if latencies else "-",
        "median latency (s)": f"{statistics.median(latencies):.1f}" if latencies else "-",
        "failed": str(sum(r.error is not None and r.refused is None and r.best_score is None for r in results)),
    }


def _summary_markdown(
    results: Sequence[QuestionResult],
    config: Config,
    retrieval_only: bool,
    suggestion: ThresholdSuggestion | None,
) -> str:
    rows = summarise(results)
    lines = [
        "# Evaluation run",
        "",
        f"- Run at: {datetime.now().isoformat(timespec='seconds')}",
        f"- Mode: {'search only (the answer model was not used)' if retrieval_only else 'search and answer'}",
        f"- Questions: {len(results)}",
        f"- Answer model: {config.llm.answer_model} (prompt version {PROMPT_VERSION})",
        f"- Embedding model: {config.embedding.model}",
        f"- Reranker: {config.reranker.model} (normalised scores: {config.reranker.normalize}, "
        f"8-bit weights: {config.reranker.quantize})",
        f"- Candidates: {config.retrieval.candidates}, kept: {config.retrieval.top_k}",
        f"- Rerank threshold: {config.retrieval.rerank_score_threshold}",
        "",
        "Each cell is `passed/counted`. A question is counted only where the measure "
        "applies to it: `-` means it applied to none.",
        "",
    ]
    if rows:
        headers = list(rows[0])
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join("---" for _ in headers) + "|")
        lines.extend("| " + " | ".join(row[header] for header in headers) + " |" for row in rows)
    if suggestion is not None:
        lines += ["", "## Suggested threshold", "", *_suggestion_lines(suggestion)]
    lines += [
        "",
        "Whether each answer is right is not judged here. Fill in `manual_correct` in "
        "`results.csv`.",
        "",
    ]
    return "\n".join(lines)


def _suggestion_lines(suggestion: ThresholdSuggestion) -> list[str]:
    return [
        f"retrieval.rerank_score_threshold: {suggestion.threshold}",
        f"With it, {suggestion.answerable_kept} of {suggestion.answerable_total} answerable "
        f"question(s) are kept and {suggestion.unanswerable_refused} of "
        f"{suggestion.unanswerable_total} {NOT_IN_DOCS} question(s) are refused.",
        "It is drawn from these questions only. With few questions it is a rough start: "
        "look at the scores in results.csv before you rely on it.",
    ]


def _page_found(question: Question, ranked: Sequence[ScoredChunk], doc_ids: set[str]) -> bool | None:
    if not question.answerable or question.page is None:
        return None
    return any(
        item.chunk.doc_id in doc_ids and item.chunk.covers_page(question.page) for item in ranked[:TOP_N]
    )


def _passages(ranked: Sequence[ScoredChunk]) -> tuple[str, ...]:
    return tuple(
        f"{item.chunk.doc_title} p. {item.chunk.page}"
        + (f"-{item.chunk.last_page}" if item.chunk.last_page != item.chunk.page else "")
        + (f" ({item.rerank_score:.2f})" if item.rerank_score is not None else "")
        for item in ranked[:TOP_N]
    )


def _not_ingested(question: Question, doc_ids: set[str]) -> str | None:
    if question.source_doc and not doc_ids:
        return f"source_doc {question.source_doc!r} is not among the ingested documents"
    return None


def _rate(values: Sequence[bool | None]) -> str:
    counted = [value for value in values if value is not None]
    return f"{sum(counted)}/{len(counted)}" if counted else "-"


def _flag(value: bool | None) -> str:
    return "" if value is None else ("yes" if value else "no")


# -- command line ---------------------------------------------------------------------


def with_answer_model(config: Config, name: str) -> Config:
    """The configuration with another answer model, and everything else as it is.

    For comparing models on the same questions without editing ``config.yaml``: run
    the evaluation once as configured and once with ``--answer-model``, then put the
    two ``results.csv`` files side by side. Each run's ``summary.md`` names its model.
    """
    return config.model_copy(update={"llm": config.llm.model_copy(update={"answer_model": name})})


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m docqa.evaluate",
        description="Run the test questions and report the results per question type.",
    )
    parser.add_argument("--config", help="path to config.yaml (default: the project's own)")
    parser.add_argument("--questions", help="question file (default: paths.eval_questions)")
    parser.add_argument("--limit", type=int, metavar="N", help="run only the first N questions")
    parser.add_argument("--type", action="append", metavar="TYPE", help="run only this type; repeatable")
    parser.add_argument(
        "--retrieval-only", action="store_true", help="judge the search only; the answer model is not used"
    )
    parser.add_argument(
        "--suggest-threshold",
        action="store_true",
        help="propose retrieval.rerank_score_threshold from the questions (implies --retrieval-only)",
    )
    parser.add_argument("--out", help="folder for the results (default: a new one in paths.eval_results_dir)")
    parser.add_argument(
        "--answer-model",
        metavar="NAME",
        help="answer with this model instead of llm.answer_model, to compare two models "
        "on the same questions; the model must be on the model server",
    )
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    retrieval_only = args.retrieval_only or args.suggest_threshold

    try:
        config = load_config(args.config)
        if args.answer_model:
            config = with_answer_model(config, args.answer_model)
        questions = load_questions(Path(args.questions) if args.questions else config.paths.eval_questions)
        if args.type:
            wanted = {kind.lower() for kind in args.type}
            questions = [question for question in questions if question.type in wanted]
        if args.limit is not None:
            questions = questions[: args.limit]
        if not questions:
            raise EvaluationError(
                "There are no questions to run. Add rows to the question file; eval/README.md "
                "says what goes in each column."
            )
        results = evaluate(questions, config, retrieval_only=retrieval_only, progress=print)
    except (ConfigError, EvaluationError, StoreError, EmbeddingError, RerankError) as exc:
        print(f"The evaluation could not run: {exc}")
        return 1
    finally:
        resources.close_all()

    suggestion = suggest_threshold(results) if args.suggest_threshold else None
    folder = Path(args.out) if args.out else config.paths.eval_results_dir / datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )
    write_results(results, config, folder, retrieval_only=retrieval_only, suggestion=suggestion)

    print()
    for row in summarise(results):
        print("  ".join(f"{key}: {value}" for key, value in row.items()))
    if args.suggest_threshold:
        print()
        if suggestion is None:
            print(
                f"No threshold can be suggested: the questions must include both answerable "
                f"ones and {NOT_IN_DOCS} ones."
            )
        else:
            print("\n".join(_suggestion_lines(suggestion)))
    print(f"\nResults written to {folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
