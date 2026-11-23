"""The chat screen.

Run from the project folder::

    streamlit run src/docqa/app/main.py

What it does with a question:

1. searches the indexed documents (the first question also loads the search models,
   which takes a while);
2. if nothing relevant enough is found, replies with the fixed not-found sentence;
3. otherwise shows the answer while the model writes it, then the citations;
4. lists every passage that was retrieved, each one expandable: a table is shown as
   the original table, a figure as its cropped image;
5. records the question, the passages with their scores, the answer, the time taken
   and the model names in the query log.

Each question is answered on its own. A follow-up such as "and for personal loans?"
is not understood as one: earlier questions are not passed to the model.

Stop this app before ingesting documents. Both use the same index, and Qdrant's local
mode lets only one program open it.
"""

from __future__ import annotations

import time
from typing import Any

import streamlit as st

from docqa import resources
from docqa.config import Config, ConfigError, load_config
from docqa.generate import answer, log_query
from docqa.index import count_chunks
from docqa.index.embedder import EmbeddingError
from docqa.index.store import StoreError
from docqa.llm import LLMError
from docqa.retrieve import retrieve
from docqa.retrieve.reranker import RerankError
from docqa.schema import Answer, ElementType, ScoredChunk

PAGE_TITLE = "BFSI document Q&A"
NO_DOCUMENTS_MESSAGE = "No documents are indexed yet."
INGEST_HINT = (
    "To index some: stop this app, run `python -m docqa.ingest docs/ --limit 3` from the "
    "project folder, then start the app again."
)
THRESHOLD_MESSAGE = (
    "`retrieval.rerank_score_threshold` is not set in `config.yaml`, so the chatbot cannot "
    "decide when to say that it found nothing. Stop this app, run "
    "`python -m docqa.evaluate --suggest-threshold` on your test questions, put the value "
    "it proposes in `config.yaml`, and start the app again."
)
FIGURE_WARNING = (
    "This description was written by the vision model. It may contain mistakes: check it "
    "against the figure."
)
_FAILURES = (LLMError, EmbeddingError, RerankError, StoreError, ConfigError)


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE)
    st.title(PAGE_TITLE)

    try:
        config = load_config()
    except ConfigError as exc:
        st.error(f"The configuration could not be loaded.\n\n{exc}")
        return

    st.caption(
        f"Configured models: answers `{config.llm.answer_model}`, "
        f"figures `{config.llm.vision_model}`, embeddings `{config.embedding.model}`, "
        f"reranker `{config.reranker.model}`."
    )

    try:
        chunks = count_chunks(config)
    except StoreError as exc:
        st.error(str(exc))
        return
    if chunks == 0:
        st.info(f"{NO_DOCUMENTS_MESSAGE} {INGEST_HINT}")
        return

    _sidebar(config, chunks)
    if config.retrieval.rerank_score_threshold is None:
        st.warning(THRESHOLD_MESSAGE)
        return

    history: list[dict[str, Any]] = st.session_state.setdefault("history", [])
    for turn in history:
        with st.chat_message("user"):
            st.markdown(_safe(turn["question"]))
        with st.chat_message("assistant"):
            if turn["error"]:
                st.error(turn["error"])
            else:
                st.markdown(_safe(turn["answer"].text))
                _details(turn["answer"], config)

    question = st.chat_input("Ask a question about the documents")
    if question and question.strip():
        _ask(question.strip(), config, history)


def _ask(question: str, config: Config, history: list[dict[str, Any]]) -> None:
    """Answer one question on screen, log it, and add it to the conversation."""
    with st.chat_message("user"):
        st.markdown(_safe(question))

    with st.chat_message("assistant"):
        started = time.perf_counter()
        try:
            with st.status(
                "Searching the documents. The first question also loads the search models.",
                expanded=False,
            ) as status:
                retrieval = retrieve(question, config)
                status.update(
                    label=(
                        f"Searched in {retrieval.elapsed_s:.0f} s. "
                        + (
                            f"Reading {len(retrieval.chunks)} passage(s)."
                            if retrieval.found
                            else "Nothing relevant enough was found."
                        )
                    ),
                    state="complete",
                )
            placeholder = st.empty()
            written: list[str] = []

            def show(piece: str) -> None:
                written.append(piece)
                placeholder.markdown(_safe("".join(written)) + " ▌")

            result = answer(question, retrieval, resources.get_llm(config), config, on_text=show)
            placeholder.markdown(_safe(result.text))
            log_query(result, config)
        except _FAILURES as exc:
            message = f"{type(exc).__name__}: {exc}"
            st.error(message)
            _log_failure(question, message, time.perf_counter() - started, config)
            history.append({"question": question, "answer": None, "error": message})
            return
        _details(result, config)
    history.append({"question": question, "answer": result, "error": None})


def _details(result: Answer, config: Config) -> None:
    """Under an answer: how long it took, what it cites, and the passages it was given."""
    took = f"{result.latency_s:.0f} s"
    if result.answer_model:
        st.caption(f"Answered in {took} by `{result.answer_model}`.")
    else:
        st.caption(f"Answered in {took}. The answer model was not called.")

    if result.citations:
        st.markdown(
            "**Cited:** "
            + "; ".join(_safe(f"[{c.doc_title}, p. {c.page}]") for c in result.citations)
        )
    if not result.sources:
        return

    if result.refused:
        best = result.sources[0].rerank_score
        threshold = config.retrieval.rerank_score_threshold
        if best is not None and threshold is not None and best < threshold:
            st.caption(
                f"Closest passages found. They were not used: the best score, {best:.2f}, is "
                f"below the threshold of {threshold:.2f}."
            )
        else:
            st.caption("Passages that were given to the model:")
    else:
        st.caption("Passages given to the model, best first. Cited ones are marked.")

    cited = {citation.chunk_id for citation in result.citations}
    for number, source in enumerate(result.sources, start=1):
        with st.expander(_source_label(number, source, source.chunk.chunk_id in cited)):
            _source(source)


def _source_label(number: int, source: ScoredChunk, cited: bool) -> str:
    chunk = source.chunk
    pages = f"p. {chunk.page}" if chunk.last_page == chunk.page else f"pp. {chunk.page}-{chunk.last_page}"
    kind = {
        ElementType.TEXT: "text",
        ElementType.TABLE: "table",
        ElementType.FIGURE: "figure",
    }[chunk.element_type]
    score = f", score {source.rerank_score:.2f}" if source.rerank_score is not None else ""
    return _safe(f"{number}. {chunk.doc_title}, {pages} ({kind}{score}){' - cited' if cited else ''}")


def _source(source: ScoredChunk) -> None:
    """One retrieved passage, in the form that suits it."""
    chunk = source.chunk
    facts = []
    if chunk.heading_path:
        facts.append("Section: " + " > ".join(chunk.heading_path))
    facts.append(
        f"Effective date: {chunk.effective_date.isoformat() if chunk.effective_date else 'not recorded'}"
    )
    if chunk.version:
        facts.append(f"Version: {chunk.version}")
    st.caption(_safe(" | ".join(facts)))

    if chunk.element_type is ElementType.TABLE and chunk.html:
        st.html(_bordered(chunk.html))
    elif chunk.element_type is ElementType.FIGURE:
        if chunk.image_path is not None and chunk.image_path.is_file():
            st.image(str(chunk.image_path))
        elif chunk.image_path is not None:
            st.caption(f"The image file is missing: {chunk.image_path}")
        if chunk.model_generated:
            st.warning(FIGURE_WARNING)
        st.text(chunk.text)
    else:
        st.text(chunk.text)


def _sidebar(config: Config, chunks: int) -> None:
    documents = resources.get_database(config).list_documents()
    with st.sidebar:
        st.subheader("Indexed documents")
        st.caption(f"{len(documents)} document(s), {chunks} passage(s).")
        for record in documents:
            when = record.effective_date.isoformat() if record.effective_date else "no date"
            st.markdown(_safe(f"- {record.doc_title} ({when})"))
        st.divider()
        st.caption(
            "Each question is answered on its own. Earlier questions are not taken into "
            "account. Every answer has to be checked against the pages it cites."
        )


def _log_failure(question: str, message: str, seconds: float, config: Config) -> None:
    try:
        resources.get_database(config).log_failure(
            question,
            message,
            seconds,
            config.embedding.model,
            config.reranker.model,
            config.llm.answer_model,
        )
    except Exception:  # noqa: BLE001 - a logging problem must not hide the original error
        pass


def _safe(text: str) -> str:
    """Text for ``st.markdown``: a dollar sign must not start a formula."""
    return text.replace("$", "\\$")


def _bordered(table_html: str) -> str:
    """Docling's table HTML carries no styling. Give the table visible cell borders."""
    return table_html.replace(
        "<table>", '<table border="1" cellpadding="4" style="border-collapse: collapse;">', 1
    )


if __name__ == "__main__":
    main()
