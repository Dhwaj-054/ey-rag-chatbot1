"""Ingestion started from the chat screen ("Add documents").

The page can add PDFs in two ways:

- by sending files from the computer it is opened on. They are stored in
  ``paths.uploads_dir`` and named ``uploads/<file>`` in the register, so that they can
  never be taken for a file of the documents folder;
- by ticking files that already lie in ``paths.docs_dir`` and were never ingested.

Either way the files then go through the same pipeline as ``python -m docqa.ingest``
(``docqa.ingest.pipeline.ingest_path``), in a thread of the server. Running it inside
the server is what lets it share the index with the questions: Qdrant's local mode
lets one program open the index, so a second program could not.

One ingestion runs at a time, and no question is answered while it runs: both would
want the whole processor. Before it starts, the search models are dropped from memory,
because parsing and describing figures need the room; they are loaded again when it
ends. The page asks for the state of the run once a second (``state_payload``) and can
ask it to stop, which takes effect when the step of the file in hand is done.

The same check for personal data applies as on the command line: a file that looks as
if it holds PAN or Aadhaar numbers is not ingested, and the page cannot override that.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
import time
import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING, Any

from starlette.requests import Request

from docqa import resources
from docqa.config import Config
from docqa.db import Database, utc_now
from docqa.ingest.documents import (
    Manifest,
    ManifestError,
    default_title,
    doc_id_for,
    source_key,
    source_path_for,
)
from docqa.ingest.pipeline import find_documents, ingest_path
from docqa.schema import IngestReport
from docqa.stats import short_title

if TYPE_CHECKING:
    from docqa.web.server import AppState

log = logging.getLogger(__name__)

_MAX_LINES = 2000
_MAX_NAME_LENGTH = 150
_PDF_MARK = b"%PDF-"
_WINDOWS_DEVICE_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL", *(f"COM{n}" for n in range(1, 10)), *(f"LPT{n}" for n in range(1, 10))}
)
_STEP_STATES = {"read": "reading", "figures": "figures", "index": "indexing"}
_IN_HAND = frozenset(_STEP_STATES.values())


class IngestJob:
    """One ingestion run started from the page, and what it has done so far.

    The run's thread writes to it and the server's request threads read it, so every
    access goes through its lock.
    """

    def __init__(self, number: int, describe_figures: bool, what: str, files: list[str]) -> None:
        self.number = number
        self.what = what
        """``chosen``: the files that were ticked. ``refresh``: every ingested document
        whose file or settings have changed."""
        self.describe_figures = describe_figures
        self.state = "waiting"
        """waiting (for an answer to finish) -> running -> done | failed | stopped,
        with ``stopping`` in between once a stop was asked for."""
        self.started_at = utc_now()
        self.stop = threading.Event()
        self.report: dict[str, Any] | None = None
        self.error: str | None = None
        self._began = time.monotonic()
        self._ended: float | None = None
        self._lock = threading.Lock()
        self._lines: list[str] = []
        self._dropped = 0
        self._files: dict[str, dict[str, Any]] = {}
        self._in_hand: str | None = None
        self._current: dict[str, Any] | None = None
        for source_path in files:
            self._add_file(source_path)

    @property
    def active(self) -> bool:
        return self.state in ("waiting", "running", "stopping")

    # -- written by the run -------------------------------------------------------------

    def begin(self) -> None:
        with self._lock:
            if self.state == "waiting":
                self.state = "running"

    def say(self, line: str) -> None:
        with self._lock:
            self._lines.append(line)
            if len(self._lines) > _MAX_LINES:
                extra = len(self._lines) - _MAX_LINES
                del self._lines[:extra]
                self._dropped += extra

    def on_event(self, kind: str, data: dict[str, Any]) -> None:
        """What the pipeline reports (``ingest_path``'s ``on_event``)."""
        with self._lock:
            if kind == "plan":
                for source_path in data["files"]:
                    self._add_file(source_path)
            elif kind == "file":
                key = source_key(data["source_path"])
                self._put_down(key)
                entry = self._add_file(data["source_path"])
                entry["state"] = _STEP_STATES[data["step"]]
                self._in_hand = key
                self._current = {
                    "source_path": entry["source_path"],
                    "name": entry["name"],
                    "step": data["step"],
                    "position": list(self._files).index(key) + 1,
                    "of": len(self._files),
                }
            elif kind == "figure":
                if self._current is not None:
                    self._current["figure"] = {
                        "position": data["position"],
                        "of": data["of"],
                        "page": data["page"],
                    }
            elif kind == "file_done":
                entry = self._add_file(data["source_path"])
                entry["state"] = "done"
                pages = data.get("pages")
                entry["detail"] = f"{data['chunks']} passage(s)" + (f", {pages} page(s)" if pages else "")
                entry["notes"] = list(data.get("notes") or [])
            elif kind == "file_failed":
                entry = self._add_file(data["source_path"])
                entry["state"] = "failed"
                entry["detail"] = data["problem"]

    def show_step(self, step: str) -> None:
        """A step of the run that is not about one file (loading the search models again)."""
        with self._lock:
            self._put_down(None)
            self._current = {"step": step}

    def finish(self, state: str, report: dict[str, Any] | None, error: str | None = None) -> None:
        with self._lock:
            self._put_down(None)
            self._current = None
            for entry in self._files.values():
                if entry["state"] not in ("done", "failed"):
                    entry["state"] = "left"  # the run ended before this file was through
            self.report = report
            self.error = error
            self.state = state
            self._ended = time.monotonic()

    def ask_to_stop(self) -> None:
        self.stop.set()
        with self._lock:
            if self.state in ("waiting", "running"):
                self.state = "stopping"

    # -- read by the page ---------------------------------------------------------------

    def snapshot(self, since: int = 0) -> dict[str, Any]:
        """The run as the page is told it. ``since``: how many log lines the page has already."""
        with self._lock:
            files = [dict(entry) for entry in self._files.values()]
            return {
                "number": self.number,
                "state": self.state,
                "what": self.what,
                "describe_figures": self.describe_figures,
                "started_at": self.started_at,
                "elapsed_s": (self._ended if self._ended is not None else time.monotonic()) - self._began,
                "total": len(files),
                "done": sum(1 for entry in files if entry["state"] == "done"),
                "failed": sum(1 for entry in files if entry["state"] == "failed"),
                "current": dict(self._current) if self._current else None,
                "files": files,
                "lines": self._lines[max(since - self._dropped, 0) :],
                "line_count": self._dropped + len(self._lines),
                "report": self.report,
                "error": self.error,
            }

    # -- internals (call with the lock held) ---------------------------------------------

    def _add_file(self, source_path: str) -> dict[str, Any]:
        key = source_key(source_path)
        if key not in self._files:
            self._files[key] = {
                "source_path": source_path,
                "name": source_path.rsplit("/", 1)[-1],
                "state": "waiting",
                "detail": "",
            }
        return self._files[key]

    def _put_down(self, next_key: str | None) -> None:
        """The file that was being worked on now waits for its next step, unless it is finished."""
        if self._in_hand is not None and self._in_hand != next_key:
            entry = self._files.get(self._in_hand)
            if entry is not None and entry["state"] in _IN_HAND:
                entry["state"] = "between"
        self._in_hand = None


# -- what is waiting ------------------------------------------------------------------------


def roots(config: Config) -> dict[str, tuple[Path, Path]]:
    """The folders the page can ingest from: name -> (folder, folder its files are named below).

    Uploads come first: they are what the person at the page has just added.
    """
    return {
        "uploads": (config.paths.uploads_dir, config.paths.uploads_dir.parent),
        "docs": (config.paths.docs_dir, config.paths.docs_dir),
    }


def waiting_files(config: Config, db: Database) -> list[dict[str, Any]]:
    """The PDFs in the uploads folder and the documents folder that were never ingested.

    No file is opened for this: a file counts as ingested if the register has a
    document of its name. A file that was ingested and has changed since is not
    listed; "Bring the ingested documents up to date" finds those.

    Raises:
        ManifestError: The manifest cannot be read.
    """
    known = {record.doc_id for record in db.list_documents()}
    manifest = Manifest.load(config.paths.manifest)
    seen: set[str] = set()
    found = []
    for root, (folder, source_root) in roots(config).items():
        if not folder.is_dir():
            continue
        for file in find_documents(folder):
            source_path = source_path_for(file, source_root)
            key = source_key(source_path)
            if key in seen or doc_id_for(source_path) in known:
                continue
            seen.add(key)
            entry = manifest.lookup(source_path)
            title = entry.doc_title if entry is not None and entry.doc_title else default_title(file)
            found.append(
                {
                    "root": root,
                    "source_path": source_path,
                    "name": file.name,
                    "title": title,
                    "short_title": short_title(title),
                    "size": file.stat().st_size,
                }
            )
    return found


def state_payload(state: AppState, since: int = 0, with_files: bool = True) -> dict[str, Any]:
    """What ``GET /api/ingest`` answers: the run, and (unless left out) what is waiting."""
    config = state.config
    job = state.job
    payload: dict[str, Any] = {"job": job.snapshot(since) if job is not None else None}
    if with_files:
        db = resources.get_database(config)
        problem = None
        try:
            waiting = waiting_files(config, db)
        except (ManifestError, OSError) as exc:
            waiting, problem = [], str(exc)
        payload.update(
            waiting=waiting,
            problem=problem,
            folders={name: str(folder) for name, (folder, _root) in roots(config).items()},
            confirm_above=config.ingestion.confirm_above_files,
            describe_figures=config.figures.describe,
            max_upload_mb=config.ui.max_upload_mb,
        )
    return payload


# -- starting and running -------------------------------------------------------------------


def start(state: AppState, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """Start an ingestion run. Returns the HTTP status and the reply.

    ``body`` is either ``{"files": [source paths], "describe_figures": bool,
    "confirmed": bool}`` or ``{"refresh": true}``. Only files that are waiting to be
    ingested right now are accepted, so the page cannot name a file anywhere else.
    """
    config = state.config
    refresh = body.get("refresh") is True
    describe = body.get("describe_figures")
    if not isinstance(describe, bool):
        describe = config.figures.describe

    runs: list[dict[str, Any]] = []
    chosen: list[str] = []
    if refresh:
        for folder, source_root in roots(config).values():
            if folder.is_dir():
                runs.append({"folder": folder, "source_root": source_root, "only": None, "known_only": True})
        if not runs:
            return 400, {"error": "Neither the documents folder nor the uploads folder exists."}
    else:
        wanted = body.get("files")
        if not isinstance(wanted, list) or not wanted or not all(isinstance(name, str) for name in wanted):
            return 400, {"error": "Name the files to ingest: {\"files\": [...]}."}
        try:
            available = {
                source_key(item["source_path"]): item
                for item in waiting_files(config, resources.get_database(config))
            }
        except (ManifestError, OSError) as exc:
            return 400, {"error": str(exc)}
        unknown = [name for name in wanted if source_key(name) not in available]
        if unknown:
            return 400, {
                "error": f"{len(unknown)} of the files named are not waiting to be ingested "
                f"(for example {unknown[0]!r}). Reload the list and choose again."
            }
        by_root: dict[str, list[str]] = {}
        for name in dict.fromkeys(source_key(name) for name in wanted):
            item = available[name]
            by_root.setdefault(item["root"], []).append(item["source_path"])
            chosen.append(item["source_path"])
        limit = config.ingestion.confirm_above_files
        if len(chosen) > limit and body.get("confirmed") is not True:
            return 409, {
                "error": f"{len(chosen)} files were chosen, which is more than {limit} "
                f"(ingestion.confirm_above_files). Parsing and embedding are slow on a "
                f"processor: this can take hours. Confirm to go ahead.",
                "needs_confirmation": True,
                "files": len(chosen),
            }
        for root, (folder, source_root) in roots(config).items():
            if root in by_root:
                runs.append(
                    {"folder": folder, "source_root": source_root, "only": by_root[root], "known_only": False}
                )

    with state.job_lock:
        if state.ingesting():
            return 409, {"error": "An ingestion is already running."}
        number = state.job.number + 1 if state.job is not None else 1
        job = IngestJob(number, describe, "refresh" if refresh else "chosen", chosen)
        state.job = job
    threading.Thread(target=run, args=(state, job, runs), name="docqa-ingest", daemon=True).start()
    return 202, {"job": job.snapshot(0)}


def run(state: AppState, job: IngestJob, runs: list[dict[str, Any]]) -> None:
    """The ingestion run itself. Runs in its own thread until it is done or stopped."""
    config = state.config
    with state.busy:  # waits here while an answer is being written
        if job.stop.is_set():
            job.finish("stopped", merged([]))
            return
        job.begin()
        reports: list[IngestReport] = []
        outcome, error = "done", None
        try:
            # Parsing and describing figures need the memory the search models hold.
            resources.release("reranker")
            resources.release("embedder")
            for one in runs:
                report = ingest_path(
                    one["folder"],
                    config,
                    describe_figures=job.describe_figures,
                    assume_yes=True,  # the page has asked before a large run
                    known_only=one["known_only"],
                    only=one["only"],
                    source_root=one["source_root"],
                    should_stop=job.stop.is_set,
                    on_event=job.on_event,
                    progress=job.say,
                )
                reports.append(report)
                if report.stopped:
                    outcome = "stopped"
                    break
        except Exception as exc:  # noqa: BLE001 - shown on the page, not raised
            log.exception("The ingestion started from the chat screen failed.")
            outcome, error = "failed", f"{type(exc).__name__}: {exc}"
            job.say(f"FAILED: {error}")
        finally:
            if state.warm_up_wanted:
                job.show_step("models")
                job.say("Loading the search models again, for questions.")
                state.load_search_models()
            job.finish(outcome, merged(reports), error)


def merged(reports: list[IngestReport]) -> dict[str, Any]:
    """The reports of the runs over each folder, added up, as the page is told them."""
    return {
        "files_seen": sum(report.files_seen for report in reports),
        "files_skipped": sum(report.files_skipped for report in reports),
        "files_to_ingest": sum(report.files_to_ingest for report in reports),
        "files_ingested": sum(report.files_ingested for report in reports),
        "chunks_indexed": sum(report.chunks_indexed for report in reports),
        "figures_described": sum(report.figures_described for report in reports),
        "failures": [line for report in reports for line in report.failures],
        "warnings": [line for report in reports for line in report.warnings],
        "read_s": sum(report.read_s for report in reports),
        "figures_s": sum(report.figures_s for report in reports),
        "index_s": sum(report.index_s for report in reports),
        "stopped": any(report.stopped for report in reports),
    }


# -- uploads ------------------------------------------------------------------------------


def safe_pdf_name(name: str) -> str | None:
    """A file name that is safe to store an upload under, or ``None`` if it is not a PDF's.

    Only the last part of what the browser sent is used, so the name cannot point
    outside the uploads folder. Letters and digits of any script are kept; anything
    that could mean something to the file system becomes an underscore.
    """
    name = name.replace("\\", "/").rsplit("/", 1)[-1].strip()
    if not name.lower().endswith(".pdf"):
        return None
    # Letters, marks and digits of any script: a Hindi name keeps its vowel signs, which
    # are marks and not letters.
    stem = "".join(
        character if unicodedata.category(character)[0] in "LMN" or character in " ._()-,&" else "_"
        for character in name[:-4]
    )
    stem = re.sub(r"\s+", " ", stem).strip(" .")
    if not stem:
        return None
    if stem.split(".")[0].upper() in _WINDOWS_DEVICE_NAMES:
        stem = f"_{stem}"
    return f"{stem[:_MAX_NAME_LENGTH]}.pdf"


async def receive_upload(request: Request, config: Config) -> tuple[int, dict[str, Any]]:
    """Store one uploaded PDF in ``paths.uploads_dir``. Returns the HTTP status and the reply.

    The file is the body of the request; its name is the ``name`` query parameter. It
    is written to a temporary file first and moved into place once it is known to be
    whole, to be a PDF, and not to be a document that is ingested already.
    """
    name = safe_pdf_name(request.query_params.get("name", ""))
    if name is None:
        return 400, {"error": "Only PDF files can be added (the name has to end in .pdf)."}
    limit = config.ui.max_upload_mb * 1024 * 1024
    too_large = {
        "error": f"The file is larger than {config.ui.max_upload_mb} MB (ui.max_upload_mb in config.yaml)."
    }
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        return 413, too_large

    folder = config.paths.uploads_dir
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / f".{name}.{threading.get_ident()}.part"
    digest = hashlib.sha256()
    size = 0
    start_of_file = b""
    try:
        with open(temporary, "wb") as handle:
            async for block in request.stream():
                size += len(block)
                if size > limit:
                    break
                if len(start_of_file) < 1024:
                    start_of_file += block[: 1024 - len(start_of_file)]
                digest.update(block)
                handle.write(block)
        if size > limit:
            return 413, too_large
        if size == 0:
            return 400, {"error": f"{name} is empty."}
        if _PDF_MARK not in start_of_file:
            return 400, {"error": f"{name} is not a PDF file."}
        existing = resources.get_database(config).find_document_by_hash(digest.hexdigest())
        if existing is not None:
            return 200, {
                "status": "duplicate",
                "name": name,
                "title": existing.doc_title,
                "source_path": existing.source_path,
            }
        os.replace(temporary, folder / name)
    finally:
        temporary.unlink(missing_ok=True)
    return 200, {
        "status": "stored",
        "name": name,
        "size": size,
        "source_path": source_path_for(folder / name, folder.parent),
    }
