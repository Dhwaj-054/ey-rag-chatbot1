"""Ingestion from start to finish: one file, or every PDF in a folder.

    hash -> skip if unchanged -> read (parse, cached; check for personal data)
         -> describe figures (cached) -> chunk, embed and store -> record

A file is skipped when neither its content nor anything that shapes its chunks has
changed since it was last ingested. "Anything that shapes its chunks" is the
fingerprint: the document's title, version, date and access group, and the parsing,
figure, chunking and embedding settings. So correcting a title in the manifest, or
changing ``chunking.max_tokens``, re-ingests the documents it affects without anyone
having to remember to. Parsing and figure descriptions come from their caches in that
case, which leaves the embedding as the only real work.

**The work is done in three steps, a group of files at a time**
(``ingestion.group_files``), and not file by file:

    step 1  read every file of the group        (Docling's models in memory)
    step 2  describe every figure of the group  (the vision model, in Ollama)
    step 3  chunk, embed and store the group    (the embedding model in memory)

Each step needs a different large model. Taking the steps in turn means only one of
them is in memory at a time, which matters on a machine with 16 GB: with all three
loaded at once, Windows starts swapping to disk and everything slows down. After each
group its documents can be searched, so a long run that is interrupted has not been
for nothing, and the caches make the next run pick up where it stopped.

One file that fails does not stop the others. Its message goes into the report.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from docqa.config import Config
from docqa.db import Database, DocumentRecord, utc_now
from docqa.index import index_chunks, searchable_text
from docqa.index.embedder import Embedder
from docqa.index.store import ChunkStore
from docqa.ingest.chunker import chunk_elements
from docqa.ingest.documents import (
    Manifest,
    doc_id_for,
    document_info,
    file_sha256,
    source_path_for,
)
from docqa.ingest.elements import Extraction, extract_elements
from docqa.ingest.figures import PROMPT_VERSION, describe_figure_cached
from docqa.ingest.parser import cache_paths, load_parsed, parse_settings, release_converters
from docqa.ingest.screening import screen_elements
from docqa.llm import LLMClient
from docqa.schema import DocumentInfo, ElementType, FigureDescription, IngestReport
from docqa.tokens import TokenCounter

log = logging.getLogger(__name__)

# Raise this when the chunking or indexing logic changes in a way that should make every
# document be ingested again. It is part of each document's fingerprint.
PIPELINE_VERSION = "2"

DOCUMENT_SUFFIXES = frozenset({".pdf"})


class IngestAborted(Exception):
    """The run would process more files than ``ingestion.confirm_above_files`` allows unasked."""


class FlaggedDocument(Exception):
    """The file looks as if it holds personal data and was not processed."""


@dataclass(frozen=True)
class _Work:
    path: Path
    source_path: str
    info: DocumentInfo
    fingerprint: str
    reason: str


@dataclass
class _Read:
    """One file after step 1: what was found in it, waiting for steps 2 and 3."""

    item: _Work
    extraction: Extraction
    notes: list[str]
    descriptions: dict[str, FigureDescription] = field(default_factory=dict)


def find_documents(path: Path) -> list[Path]:
    """The documents at ``path``: the file itself, or every PDF below the folder.

    Only PDFs are taken. A sample folder also holds other things (a README, a diagram),
    and Docling would happily read an image as if it were a document.
    """
    if path.is_file():
        return [path] if path.suffix.lower() in DOCUMENT_SUFFIXES else []
    if not path.is_dir():
        raise FileNotFoundError(f"No such file or folder: {path}")
    return sorted(
        (item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in DOCUMENT_SUFFIXES),
        key=lambda item: item.as_posix().lower(),
    )


def fingerprint(info: DocumentInfo, config: Config, describe: bool) -> str:
    """A hash of everything, other than the file itself, that shapes a document's chunks."""
    figures = config.figures
    parts = {
        "pipeline_version": PIPELINE_VERSION,
        "doc_title": info.doc_title,
        "version": info.version,
        "effective_date": info.effective_date.isoformat() if info.effective_date else None,
        "access_group": info.access_group,
        "parsing": parse_settings(config),
        "chunking": config.chunking.model_dump(),
        "figures": {
            "describe": describe,
            "min_page_fraction": figures.min_page_fraction,
            "min_side_px": figures.min_side_px,
            "max_aspect_ratio": figures.max_aspect_ratio,
            "vision_model": config.llm.vision_model if describe else None,
            "prompt_version": PROMPT_VERSION if describe else None,
        },
        "embedding": {
            "model": config.embedding.model,
            "max_length": config.embedding.max_length,
            "quantize": config.embedding.quantize,
        },
    }
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode("utf-8")).hexdigest()


def ingest_path(
    path: Path,
    config: Config,
    llm: LLMClient | None = None,
    *,
    limit: int | None = None,
    force: bool = False,
    assume_yes: bool = False,
    allow_flagged: bool = False,
    describe_figures: bool | None = None,
    prune: bool = False,
    known_only: bool = False,
    embedder: Embedder | None = None,
    store: ChunkStore | None = None,
    db: Database | None = None,
    counter: TokenCounter | None = None,
    progress: Callable[[str], None] | None = None,
) -> IngestReport:
    """Ingest one file, or every document in a folder.

    For each file: hash it and skip it if unchanged, otherwise parse, describe its
    figures, chunk, and hand the chunks to ``docqa.index.index_chunks``. The files
    themselves are never modified.

    ``llm`` is needed only when there are figures to describe. Left out, a client is
    created from the configuration when the first figure comes up.

    Args:
        limit: Process at most this many files in this run. The rest stay for the next.
        force: Ingest files even if nothing has changed.
        assume_yes: Go ahead even if more than ``ingestion.confirm_above_files`` files
            need work.
        allow_flagged: Ingest files that the personal-data check flags. Use only after
            looking at the file.
        describe_figures: Overrides ``figures.describe`` for this run.
        prune: Also remove documents whose files are no longer there.
        known_only: Leave files alone that were never ingested. Only documents already
            in the index are brought up to date, for example after a change to the
            pipeline or its settings.
        progress: Called with one line of text as each step finishes.

    Raises:
        IngestAborted: More files need work than may be processed without ``assume_yes``.
        FileNotFoundError: ``path`` does not exist.
    """
    from docqa import resources

    say = progress or log.info
    describe = config.figures.describe if describe_figures is None else describe_figures
    db = db or resources.get_database(config)
    manifest = Manifest.load(config.paths.manifest)
    files = find_documents(path)
    source_root = path.resolve() if path.is_dir() else path.parent.resolve()

    work: list[_Work] = []
    skipped = left_alone = 0
    for file in files:
        source_path = source_path_for(file, source_root)
        if known_only and db.get_document(doc_id_for(source_path)) is None:
            left_alone += 1  # not even hashed: a large folder of new files costs nothing
            continue
        sha256 = file_sha256(file)
        info = document_info(file, config, manifest, sha256, source_root)
        mark = fingerprint(info, config, describe)
        record = db.get_document(info.doc_id)
        if record is None:
            reason = "new"
        elif force:
            reason = "forced"
        elif record.sha256 != sha256:
            reason = "file changed"
        elif record.fingerprint != mark:
            reason = "settings or metadata changed"
        else:
            skipped += 1
            continue
        work.append(_Work(file, source_path_for(file, config.paths.docs_dir), info, mark, reason))

    warnings: list[str] = []
    if limit is not None and len(work) > limit:
        warnings.append(
            f"{len(work) - limit} more file(s) still need ingesting. Run the command again to go on."
        )
        work = work[:limit]
    if len(work) > config.ingestion.confirm_above_files and not assume_yes:
        raise IngestAborted(
            f"{len(work)} files need ingesting, which is more than "
            f"ingestion.confirm_above_files ({config.ingestion.confirm_above_files}). Parsing "
            f"and embedding are slow on a CPU. Use --limit N to do a few at a time, or --yes "
            f"to process them all."
        )

    never = f", {left_alone} never ingested and left alone" if known_only else ""
    say(f"{len(files)} document(s) found, {skipped} unchanged, {len(work)} to ingest{never}.")
    failures: list[str] = []
    ingested = chunks_indexed = described = 0
    seconds = {"read": 0.0, "figures": 0.0, "index": 0.0}
    own_llm: LLMClient | None = None
    loads_own_embedder = embedder is None

    def vision() -> LLMClient:
        """The client for the vision model. Created only if a figure needs describing."""
        nonlocal own_llm
        if llm is not None:
            return llm
        if own_llm is None:
            own_llm = LLMClient(config.llm.for_vision())  # ingestion only describes figures
        return own_llm

    def fail(item: _Work, exc: Exception) -> None:
        log.debug("Ingesting %s failed", item.source_path, exc_info=True)
        problem = str(exc) if isinstance(exc, FlaggedDocument) else f"{type(exc).__name__}: {exc}"
        failures.append(f"{item.source_path}: {problem}")
        say(f"    FAILED {item.source_path}: {problem}")

    try:
        if work or prune:
            # Opened before the first file, so that "the index is in use by another
            # program" stops the run at once instead of failing every file in turn.
            store = store or resources.get_store(config)
        if work:
            # The tokenizer belongs to the embedding model. Loading it now, which is
            # quick, shows at the start whether that model is on this machine, and
            # not after an hour of parsing.
            counter = counter or resources.get_token_counter(config)

        size = config.ingestion.group_files
        groups = [work[start : start + size] for start in range(0, len(work), size)]
        position = 0
        for number, group in enumerate(groups, start=1):
            if len(groups) > 1:
                say(f"Group {number} of {len(groups)}: {len(group)} document(s)")

            # Step 1: read. Only Docling's models are in memory.
            started = time.perf_counter()
            read: list[_Read] = []
            for item in group:
                position += 1
                say(f"[{position}/{len(work)}] {item.source_path} ({item.reason})")
                try:
                    read.append(_read(item, config, allow_flagged, say))
                except Exception as exc:  # noqa: BLE001 - one bad file must not stop the others
                    fail(item, exc)
            seconds["read"] += time.perf_counter() - started
            if not read:
                continue
            release_converters()

            # Step 2: describe figures. Only the vision model is at work, inside Ollama.
            if describe:
                started = time.perf_counter()
                for entry in list(read):
                    try:
                        described += _describe(entry, config, vision, db, say)
                    except Exception as exc:  # noqa: BLE001
                        fail(entry.item, exc)
                        read.remove(entry)
                seconds["figures"] += time.perf_counter() - started
            if not read:
                continue

            # Step 3: chunk, embed and store. Only the embedding model is in memory.
            started = time.perf_counter()
            group_embedder = embedder or resources.get_embedder(config)
            for entry in read:
                try:
                    count = _index(entry, config, group_embedder, store, db, counter, say)
                except Exception as exc:  # noqa: BLE001
                    fail(entry.item, exc)
                    continue
                ingested += 1
                chunks_indexed += count
                warnings.extend(f"{entry.item.source_path}: {note}" for note in entry.notes)
            seconds["index"] += time.perf_counter() - started
            if loads_own_embedder and number < len(groups):
                resources.release("embedder")  # makes room for step 1 of the next group

        if prune:
            for removed in _prune(config, store, db):
                say(f"Removed {removed}: its file is no longer there.")
    finally:
        if own_llm is not None:
            own_llm.close()

    return IngestReport(
        files_seen=len(files),
        files_skipped=skipped,
        files_ingested=ingested,
        chunks_indexed=chunks_indexed,
        failures=tuple(failures),
        warnings=tuple(warnings),
        figures_described=described,
        read_s=seconds["read"],
        figures_s=seconds["figures"],
        index_s=seconds["index"],
    )


def _read(item: _Work, config: Config, allow_flagged: bool, say: Callable[[str], None]) -> _Read:
    """Step 1 for one file: parse it (or take the cached parse) and check it for personal data."""
    info = item.info
    notes: list[str] = []

    started = time.perf_counter()
    document = load_parsed(info, config)
    extraction = extract_elements(document, info, config)
    say(
        f"    parsed: {extraction.pages} page(s), {len(extraction.elements)} element(s), "
        f"{time.perf_counter() - started:.1f} s"
    )
    if extraction.figures_without_image:
        notes.append(f"Docling returned no image for {extraction.figures_without_image} picture(s).")

    if config.screening.enabled:
        found = screen_elements(extraction.elements)
        if found.flagged(config.screening.flag_at):
            if not allow_flagged:
                raise FlaggedDocument(
                    f"not ingested, because it looks as if it may hold personal data: "
                    f"{found.describe()}. Look at the file. If it is a false alarm, run again "
                    f"with --allow-flagged."
                )
            notes.append(f"ingested although the personal-data check flagged it: {found.describe()}.")
    return _Read(item, extraction, notes)


def _describe(
    entry: _Read,
    config: Config,
    vision: Callable[[], LLMClient],
    db: Database,
    say: Callable[[str], None],
) -> int:
    """Step 2 for one file: describe its figures. Returns how many were newly described."""
    figures = [
        element
        for element in entry.extraction.elements
        if element.element_type is ElementType.FIGURE and element.image_path is not None
    ]
    if not figures:
        return 0
    newly_described = 0
    started = time.perf_counter()
    for position, element in enumerate(figures, start=1):
        figure_started = time.perf_counter()
        description, cached = describe_figure_cached(element, vision(), config, db)
        entry.descriptions[element.element_id] = description
        newly_described += not cached
        how = "cached" if cached else f"{time.perf_counter() - figure_started:.0f} s"
        say(
            f"    figure {position}/{len(figures)} of {entry.item.source_path}, p. {element.page}: "
            f"{description.figure_type} ({how})"
        )
    say(f"    figures of {entry.item.source_path}: {len(figures)}, {time.perf_counter() - started:.1f} s")
    return newly_described


def _index(
    entry: _Read,
    config: Config,
    embedder: Embedder,
    store: ChunkStore | None,
    db: Database,
    counter: TokenCounter,
    say: Callable[[str], None],
) -> int:
    """Step 3 for one file: chunk, embed, store and record it. Returns the chunks stored."""
    item, info, extraction = entry.item, entry.item.info, entry.extraction
    chunks = chunk_elements(extraction.elements, info, entry.descriptions, config, counter=counter)
    for chunk in chunks:
        if chunk.element_type is ElementType.TEXT:
            continue
        tokens = counter.count(searchable_text(chunk))
        if tokens > config.embedding.max_length:
            entry.notes.append(
                f"the {chunk.element_type.value} on p. {chunk.page} has about {tokens} tokens. "
                f"Only the first {config.embedding.max_length} are used for search "
                f"(embedding.max_length). The whole of it is stored and shown."
            )

    started = time.perf_counter()
    if chunks:
        stored = index_chunks(chunks, config, embedder=embedder, store=store)
    else:
        stored = 0
        if store is not None:
            store.delete_document(info.doc_id)
        entry.notes.append("no text, tables or figures were found in it, so nothing was indexed.")
    say(f"    indexed {item.source_path}: {stored} chunk(s), {time.perf_counter() - started:.1f} s")

    db.save_document(
        DocumentRecord(
            doc_id=info.doc_id,
            source_path=item.source_path,
            sha256=info.sha256,
            fingerprint=item.fingerprint,
            doc_title=info.doc_title,
            version=info.version,
            effective_date=info.effective_date,
            access_group=info.access_group,
            pages=extraction.pages,
            chunks=stored,
            ingested_at=utc_now(),
        )
    )
    return stored


def _prune(config: Config, store: ChunkStore | None, db: Database) -> list[str]:
    """Remove every registered document whose file no longer exists. Returns their paths."""
    removed = []
    for record in db.list_documents():
        location = Path(record.source_path)
        if not location.is_absolute():
            location = config.paths.docs_dir / location
        if location.exists():
            continue
        if store is not None:
            store.delete_document(record.doc_id)
        db.delete_document(record.doc_id)
        for leftover in cache_paths(record.doc_id, config):
            leftover.unlink(missing_ok=True)
        shutil.rmtree(config.paths.figures_dir / record.doc_id, ignore_errors=True)
        removed.append(record.source_path)
    return removed
