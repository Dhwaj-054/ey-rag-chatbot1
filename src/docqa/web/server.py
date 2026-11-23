"""The chat screen's server.

It serves one page (``static/``) and a small API for it:

    GET  /api/status      are the models loaded, is the answer model reachable, is anything indexed
    GET  /api/documents   the ingested documents, with a short name each
    GET  /api/stats       totals of the index, and typical times from the query log
    GET  /api/history     the most recent questions
    POST /api/ask         one question; the reply is a stream of events (see below)
    GET  /api/figure/<doc_id>/<file>   one cropped figure image
    GET  /api/ingest      the PDFs waiting to be ingested, and the ingestion run, if there is one
    POST /api/upload      one PDF, sent as the body of the request (?name=<file name>)
    POST /api/ingest      start ingesting the files named in the request
    POST /api/ingest/stop ask the running ingestion to stop

``/api/ask`` answers with server-sent events, so that the page can show each step as
it happens instead of waiting minutes for a finished answer. The page reads them with
the browser's Streams API (``fetch`` and ``response.body.getReader()``):

    start      {"time_limit_s": 120}   the clock of this answer starts
    step       {"name": ..., "state": "start" | "progress" | "end", ...}
               names: waiting (for another question or an ingestion), question (it is
               turned into vectors), database, ranking (with "done" and "of" after
               every batch), reading (the answer model reads the evidence) and writing
    retrieval  the passages found, with scores and which ones go to the answer model
    token      {"text": "..."}   one piece of the answer as the model writes it
    answer     the finished answer: text with citations written out, notes, timings
    error      {"message": "..."}
    done       {}

Things done here for speed:

- The search models are loaded in the background when the server starts
  (``ui.warm_up``), and the answer model is given its instructions and asked for one
  word. Ollama then has the model in memory and has already read the instructions,
  which are the same for every question, so the first question does not wait for
  either.
- The passages are sent to the page as soon as the search ends, before the answer
  model has written anything. The reader has something to read during the wait.
- An answer is planned to fit ``generation.time_limit_s`` (``docqa.generate.budget``).
- Closing the page, or pressing Stop, ends the request to the answer model. The CPU
  is not left working on an answer nobody is waiting for.
- One question is answered at a time, and none while documents are being ingested.
  Two heavy jobs at once would only make both slower.

Ingestion started from the page runs in this process, so it shares the index with the
questions (Qdrant's local mode lets one program open it). ``docqa.web.ingestion`` holds
that part.

The server listens on this machine only, unless ``ui.allow_network`` says otherwise.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import threading
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from docqa import resources
from docqa.config import Config, ConfigError, load_config
from docqa.generate import answer, answer_body, choose_evidence, log_query, measured_read_speed
from docqa.generate.budget import AnswerPlan, plan_answer
from docqa.generate.prompt import system_prompt
from docqa.llm import LLMError, is_local_url, system_message, user_message
from docqa.retrieve import retrieve
from docqa.schema import Answer, RetrievalResult, ScoredChunk
from docqa.stats import index_stats, list_documents, short_title
from docqa.web import ingestion
from docqa.web.ingestion import IngestJob

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
MAX_QUESTION_CHARS = 2000
_KEEP_ALIVE_SECONDS = 15
_LLM_CHECK_SECONDS = 20
_DOC_ID = re.compile(r"^[0-9a-f]{16}$")
_IMAGE_NAME = re.compile(r"^[A-Za-z0-9._-]+\.png$")


class Cancelled(Exception):
    """The page stopped waiting for this answer."""


class AppState:
    """What the server remembers between requests."""

    def __init__(self, config: Config, warm_up: bool) -> None:
        self.config = config
        self.warm_up_wanted = warm_up
        self.models: dict[str, Any] = {"state": "idle" if not warm_up else "loading", "detail": "", "seconds": {}}
        self.busy = threading.Lock()
        """Held by whatever heavy job is running: one answer, or one ingestion."""
        self.job: IngestJob | None = None
        self.job_lock = threading.Lock()
        self._llm_checked_at = 0.0
        self._llm_status: dict[str, Any] = {"reachable": None, "answer_model_present": None, "detail": ""}
        self._lock = threading.Lock()

    def warm_up(self) -> None:
        """Load the search models, then wake the answer model. Runs in its own thread."""
        self.load_search_models()
        if self.models["state"] != "ready":
            return
        try:
            started = time.perf_counter()
            # The instructions go with it. Ollama keeps what it has read of the last
            # request, and a question starts with the same instructions, so the model
            # does not read them again when the first question comes.
            resources.get_llm(self.config).chat(
                [
                    system_message(system_prompt(self.config.generation.not_found_message)),
                    user_message("Reply with the word OK."),
                ],
                max_tokens=1,
            )
            self.models["seconds"]["answer_model"] = time.perf_counter() - started
        except LLMError as exc:
            log.warning("The answer model could not be woken: %s", exc)

    def load_search_models(self) -> None:
        """Open the index and load the embedding model and the reranker."""
        self.models = {"state": "loading", "detail": "", "seconds": {}}
        try:
            seconds = resources.warm_up(self.config)
        except Exception as exc:  # noqa: BLE001 - shown on the page, not raised
            self.models = {"state": "failed", "detail": f"{type(exc).__name__}: {exc}", "seconds": {}}
            log.warning("The search models could not be loaded: %s", exc)
            return
        self.models = {"state": "ready", "detail": "", "seconds": seconds}

    def ingesting(self) -> bool:
        """True while an ingestion started from the page is waiting or running."""
        job = self.job
        return job is not None and job.active

    def llm_status(self) -> dict[str, Any]:
        """Whether the model server answers and has the answer model. Checked at most every 20 s."""
        with self._lock:
            if time.monotonic() - self._llm_checked_at < _LLM_CHECK_SECONDS and self._llm_checked_at:
                return self._llm_status
            try:
                llm = resources.get_llm(self.config)
                present = llm.has_model(self.config.llm.answer_model)
                self._llm_status = {
                    "reachable": True,
                    "answer_model_present": present,
                    "detail": "" if present else f"{self.config.llm.answer_model} is not on the model server.",
                }
            except LLMError as exc:
                self._llm_status = {"reachable": False, "answer_model_present": False, "detail": str(exc)}
            self._llm_checked_at = time.monotonic()
            return self._llm_status


def create_app(config: Config | None = None, *, warm_up: bool | None = None) -> Starlette:
    """Build the web application.

    Args:
        config: The configuration. Loaded from ``config.yaml`` if not given.
        warm_up: Overrides ``ui.warm_up``. Tests pass False.
    """
    config = config or load_config()
    state = AppState(config, config.ui.warm_up if warm_up is None else warm_up)

    @contextlib.asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        if state.warm_up_wanted:
            threading.Thread(target=state.warm_up, name="docqa-warm-up", daemon=True).start()
        yield
        resources.close_all()

    async def index(_request: Request) -> Response:
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    async def status(_request: Request) -> Response:
        return JSONResponse(await asyncio.to_thread(_status, state))

    async def documents(_request: Request) -> Response:
        found = await asyncio.to_thread(lambda: list_documents(resources.get_database(config)))
        return JSONResponse({"documents": [asdict(item) for item in found]})

    async def stats(_request: Request) -> Response:
        return JSONResponse(await asyncio.to_thread(_stats, state))

    async def history(request: Request) -> Response:
        try:
            limit = max(1, min(int(request.query_params.get("limit", "20")), 200))
        except ValueError:
            limit = 20
        rows = await asyncio.to_thread(lambda: resources.get_database(config).recent_queries(limit))
        return JSONResponse(
            {
                "questions": [
                    {
                        "asked_at": row.asked_at,
                        "question": row.question,
                        "answer": row.answer,
                        "refused": row.refused,
                        "error": row.error,
                        "latency_s": row.latency_s,
                    }
                    for row in rows
                ]
            }
        )

    async def figure(request: Request) -> Response:
        doc_id, name = request.path_params["doc_id"], request.path_params["name"]
        if not _DOC_ID.match(doc_id) or not _IMAGE_NAME.match(name):
            return JSONResponse({"error": "No such figure."}, status_code=404)
        path = config.paths.figures_dir / doc_id / name
        if not path.is_file():
            return JSONResponse({"error": "No such figure."}, status_code=404)
        return FileResponse(path, media_type="image/png", headers={"Cache-Control": "max-age=3600"})

    async def ask(request: Request) -> Response:
        if not same_origin(request):
            return _refused_origin()
        try:
            body = await request.json()
            question = str(body["question"]).strip()
        except (ValueError, KeyError, TypeError):
            return JSONResponse({"error": "Send JSON with a 'question'."}, status_code=400)
        if not question:
            return JSONResponse({"error": "The question is empty."}, status_code=400)
        if len(question) > MAX_QUESTION_CHARS:
            return JSONResponse(
                {"error": f"The question is longer than {MAX_QUESTION_CHARS} characters."}, status_code=400
            )

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()
        cancel = threading.Event()

        def emit(event: str, data: dict[str, Any]) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, (event, data))

        def work() -> None:
            started = time.perf_counter()
            try:
                _answer_question(state, question, emit, cancel)
            except Cancelled:
                log.info("An answer was stopped by the page.")
            except Exception as exc:  # noqa: BLE001 - every failure is shown and logged
                message = f"{type(exc).__name__}: {exc}"
                emit("error", {"message": message})
                _log_failure(config, question, message, time.perf_counter() - started)
            finally:
                emit("done", {})

        threading.Thread(target=work, name="docqa-answer", daemon=True).start()

        async def events() -> AsyncIterator[str]:
            try:
                while True:
                    try:
                        event, data = await asyncio.wait_for(queue.get(), timeout=_KEEP_ALIVE_SECONDS)
                    except TimeoutError:
                        yield ": still working\n\n"  # keeps the connection open during long waits
                        continue
                    yield f"event: {event}\ndata: {json.dumps(data)}\n\n"
                    if event == "done":
                        break
            finally:
                cancel.set()  # the page went away, or the answer is complete

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def ingest_state(request: Request) -> Response:
        try:
            since = max(0, int(request.query_params.get("since", "0")))
        except ValueError:
            since = 0
        with_files = request.query_params.get("files", "1") != "0"
        return JSONResponse(await asyncio.to_thread(ingestion.state_payload, state, since, with_files))

    async def upload(request: Request) -> Response:
        if not same_origin(request):
            return _refused_origin()
        if state.ingesting():
            return JSONResponse(
                {"error": "Documents are being ingested. Add files when that has finished."}, status_code=409
            )
        code, body = await ingestion.receive_upload(request, config)
        return JSONResponse(body, status_code=code)

    async def ingest_start(request: Request) -> Response:
        if not same_origin(request):
            return _refused_origin()
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError
        except ValueError:
            return JSONResponse({"error": "Send JSON that names the files to ingest."}, status_code=400)
        code, reply = await asyncio.to_thread(ingestion.start, state, body)
        return JSONResponse(reply, status_code=code)

    async def ingest_stop(request: Request) -> Response:
        if not same_origin(request):
            return _refused_origin()
        job = state.job
        if job is None or not job.active:
            return JSONResponse({"error": "No ingestion is running."}, status_code=409)
        job.ask_to_stop()
        return JSONResponse({"job": job.snapshot(0)})

    app = Starlette(
        routes=[
            Route("/", index),
            Route("/api/status", status),
            Route("/api/documents", documents),
            Route("/api/stats", stats),
            Route("/api/history", history),
            Route("/api/figure/{doc_id}/{name}", figure),
            Route("/api/ask", ask, methods=["POST"]),
            Route("/api/ingest", ingest_state, methods=["GET"]),
            Route("/api/ingest", ingest_start, methods=["POST"]),
            Route("/api/ingest/stop", ingest_stop, methods=["POST"]),
            Route("/api/upload", upload, methods=["POST"]),
            Mount("/static", app=StaticFiles(directory=STATIC_DIR), name="static"),
        ],
        lifespan=lifespan,
    )
    app.state.docqa = state
    return app


def same_origin(request: Request) -> bool:
    """False for a request that some other website made the browser send.

    The server listens on this machine, where any page open in the browser can reach
    it. A page from elsewhere must not be able to start an answer, an upload or an
    ingestion. Browsers say where a request comes from; a program that is not a
    browser (the tests, ``curl``) sends neither header and is let through.
    """
    site = request.headers.get("sec-fetch-site")
    if site is not None and site not in ("same-origin", "none"):
        return False
    origin = request.headers.get("origin")
    if origin is None:
        return True
    return urlsplit(origin).netloc == request.headers.get("host")


def _refused_origin() -> Response:
    return JSONResponse({"error": "This request did not come from the chat screen."}, status_code=403)


# -- answering one question ------------------------------------------------------------


def _answer_question(
    state: AppState, question: str, emit: Callable[[str, dict[str, Any]], None], cancel: threading.Event
) -> None:
    """Search, answer and log one question, reporting each step through ``emit``."""
    config = state.config
    if not state.busy.acquire(blocking=False):
        emit("step", {"name": "waiting", "state": "start", "reason": "ingestion" if state.ingesting() else "question"})
        state.busy.acquire()
        emit("step", {"name": "waiting", "state": "end"})
    try:
        if cancel.is_set():
            raise Cancelled
        started = time.perf_counter()
        emit("start", {"time_limit_s": config.generation.time_limit_s})

        def on_step(name: str, step_state: str, details: dict[str, Any]) -> None:
            if cancel.is_set():
                raise Cancelled  # between two batches of the ranking, for one
            emit("step", {"name": name, "state": step_state, **details})

        retrieval = retrieve(question, config, progress=on_step)
        plan = plan_answer(
            config, used_s=time.perf_counter() - started, read_chars_per_s=measured_read_speed(config)
        )
        given = choose_evidence(retrieval.chunks, config, plan.evidence_budget)[0] if retrieval.found else ()
        emit("retrieval", _retrieval_payload(retrieval, given, config, plan))
        if cancel.is_set():
            raise Cancelled

        def on_text(piece: str) -> None:
            if cancel.is_set():
                raise Cancelled
            emit("token", {"text": piece})

        reply = answer(
            question, retrieval, resources.get_llm(config), config, on_text=on_text, plan=plan, on_step=on_step
        )
        log_query(reply, config)
        emit("answer", _answer_payload(reply))
    finally:
        state.busy.release()


def _log_failure(config: Config, question: str, message: str, seconds: float) -> None:
    try:
        resources.get_database(config).log_failure(
            question, message, seconds, config.embedding.model, config.reranker.model, config.llm.answer_model
        )
    except Exception:  # noqa: BLE001 - a logging problem must not hide the original error
        log.exception("The failed question could not be logged.")


# -- what the page is sent --------------------------------------------------------------


def _pages(first: int, last: int) -> str:
    return f"p. {first}" if last == first else f"pp. {first}-{last}"


def _source_payload(rank: int, source: ScoredChunk, given: bool) -> dict[str, Any]:
    chunk = source.chunk
    image_url = None
    if chunk.image_path is not None:
        doc_folder, name = chunk.image_path.parent.name, chunk.image_path.name
        if _DOC_ID.match(doc_folder) and _IMAGE_NAME.match(name):
            image_url = f"/api/figure/{doc_folder}/{name}"
    return {
        "rank": rank,
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "doc_title": chunk.doc_title,
        "short_title": short_title(chunk.doc_title),
        "page": chunk.page,
        "page_end": chunk.last_page,
        "pages": _pages(chunk.page, chunk.last_page),
        "type": chunk.element_type.value,
        "heading_path": list(chunk.heading_path),
        "effective_date": chunk.effective_date.isoformat() if chunk.effective_date else None,
        "version": chunk.version,
        "score": source.rerank_score,
        "fusion_score": source.fusion_score,
        "text": chunk.text,
        "html": chunk.html,
        "image_url": image_url,
        "model_generated": chunk.model_generated,
        "given_to_model": given,
    }


def _retrieval_payload(
    retrieval: RetrievalResult,
    given: tuple[ScoredChunk, ...],
    config: Config,
    plan: AnswerPlan | None = None,
) -> dict[str, Any]:
    given_ids = {source.chunk.chunk_id for source in given}
    threshold = config.retrieval.rerank_score_threshold
    visible = [
        (rank, source)
        for rank, source in enumerate(retrieval.chunks, start=1)
        if threshold is None
        or source.rerank_score is None
        or source.rerank_score >= threshold
    ]
    return {
        "found": retrieval.found,
        "best_score": retrieval.best_score,
        "threshold": threshold,
        "elapsed_s": retrieval.elapsed_s,
        "timings": asdict(retrieval.timings),
        "plan": None
        if plan is None
        else {
            "time_limit_s": plan.time_limit_s,
            "evidence_chars": plan.evidence_chars,
            "read_chars_per_s": plan.read_chars_per_s,
            "speed_measured": plan.speed_measured,
            "shortened": plan.shortened,
        },
        "sources": [
            _source_payload(rank, source, source.chunk.chunk_id in given_ids)
            for rank, source in visible
        ],
    }


def _answer_payload(reply: Answer) -> dict[str, Any]:
    ranks = {source.chunk.chunk_id: rank for rank, source in enumerate(reply.sources, start=1)}
    timings = asdict(reply.timings)
    writing = (reply.timings.generation_s or 0.0) - (reply.timings.first_token_s or 0.0)
    timings["tokens_per_s"] = (
        reply.timings.completion_tokens / writing if reply.timings.completion_tokens and writing > 0 else None
    )
    return {
        "text": answer_body(reply),
        "notes": list(reply.notes),
        "refused": reply.refused,
        "citations": [
            {
                "doc_title": citation.doc_title,
                "short_title": short_title(citation.doc_title),
                "page": citation.page,
                "chunk_id": citation.chunk_id,
                "rank": ranks.get(citation.chunk_id),
                "label": f"[{citation.doc_title}, p. {citation.page}]",
            }
            for citation in reply.citations
        ],
        "unmatched": list(reply.unmatched_citations),
        "answer_model": reply.answer_model,
        "latency_s": reply.latency_s,
        "evidence_ids": list(reply.evidence_ids),
        "timings": timings,
    }


def _status(state: AppState) -> dict[str, Any]:
    config = state.config
    db = resources.get_database(config)
    index = index_stats(db)
    threshold = config.retrieval.rerank_score_threshold
    problems = []
    if index.passages == 0:
        problems.append(
            "No documents are indexed yet. Use 'Add documents' in the side panel. (From a "
            "terminal, with this screen stopped: 'python -m docqa.ingest docs --limit 3'.)"
        )
    if threshold is None:
        problems.append(
            "retrieval.rerank_score_threshold is not set in config.yaml, so the chatbot cannot "
            "decide when to say that it found nothing. Run "
            "'python -m docqa.evaluate --suggest-threshold' and put the value in config.yaml."
        )
    if state.models["state"] == "failed":
        problems.append(f"The search models could not be loaded. {state.models['detail']}")
    llm = state.llm_status()
    if llm["reachable"] is False or llm["answer_model_present"] is False:
        problems.append(f"The answer model cannot be used. {llm['detail']}")
    return {
        "ready": not problems,
        "problems": problems,
        "models": state.models,
        "llm": llm,
        "index": {"documents": index.documents, "passages": index.passages},
        "threshold": threshold,
        "ingesting": state.ingesting(),
        "settings": {
            "answer_model": config.llm.answer_model,
            # None when the answer model is on this machine; otherwise the server that
            # every question and its passages are sent to, so the page can say so.
            "answer_model_host": (
                None if is_local_url(config.llm.base_url) else urlsplit(config.llm.base_url).hostname
            ),
            "vision_model": config.llm.vision_model,
            "embedding_model": config.embedding.model,
            "reranker_model": config.reranker.model,
            "reranker_8_bit": config.reranker.quantize,
            "candidates": config.retrieval.candidates,
            "top_k": config.retrieval.top_k,
            "rerank_time_limit_s": config.retrieval.rerank_time_limit_s,
            "evidence_char_budget": config.generation.evidence_char_budget,
            "evidence_score_ratio": config.generation.evidence_score_ratio,
            "time_limit_s": config.generation.time_limit_s,
        },
    }


def _stats(state: AppState) -> dict[str, Any]:
    config = state.config
    db = resources.get_database(config)
    # The store is asked for counts by kind only once the models are in: opening it is
    # part of warming up, and a second program may have it open during ingestion.
    store = None
    if state.models["state"] == "ready" or not state.warm_up_wanted:
        try:
            store = resources.get_store(config)
        except Exception as exc:  # noqa: BLE001 - the totals are still worth showing
            log.info("Passages by kind are left out: %s", exc)
    return {"index": asdict(index_stats(db, store)), "queries": asdict(db.query_stats())}


def main(argv: list[str] | None = None) -> int:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(
        prog="python -m docqa.web", description="Start the chat screen of the document Q&A chatbot."
    )
    parser.add_argument("--config", help="path to config.yaml (default: the project's own)")
    parser.add_argument("--port", type=int, help="port to listen on (default: ui.port)")
    parser.add_argument("--open", action="store_true", help="open the screen in the default browser")
    parser.add_argument("--no-warm-up", action="store_true", help="do not load the models at start")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(exc)
        return 1
    port = args.port or config.ui.port
    url = f"http://{'127.0.0.1' if config.ui.host in ('127.0.0.1', 'localhost') else config.ui.host}:{port}"
    print(f"Chat screen: {url}   (Ctrl+C stops it)")
    if config.local_file is not None:
        print(f"Settings of this machine are taken from {config.local_file.name}.")
    if config.ui.warm_up and not args.no_warm_up:
        print("Loading the search models in the background. The screen says when they are ready.")
    if args.open:
        import webbrowser

        threading.Timer(1.0, webbrowser.open, args=(url,)).start()

    app = create_app(config, warm_up=False if args.no_warm_up else None)
    uvicorn.run(app, host=config.ui.host, port=port, log_level="warning")
    return 0
