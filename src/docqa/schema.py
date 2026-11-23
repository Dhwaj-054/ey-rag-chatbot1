"""The data that passes between pipeline stages.

    ingest    files            -> Element -> Chunk
    index     Chunk            -> vector store
    retrieve  question         -> RetrievalResult (ScoredChunk)
    generate  RetrievalResult  -> Answer (Citation)

These are plain, immutable records, so that each stage can be built and tested on its
own.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path


class ElementType(str, Enum):
    """What kind of content an element or chunk holds."""

    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"


class TextRole(str, Enum):
    """What part a text element plays in the document. Chunking depends on it."""

    TITLE = "title"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    FOOTNOTE = "footnote"


PageMark = tuple[int, int]
"""``(character offset, page)``: from that offset on, the text is on that page."""


@dataclass(frozen=True)
class BoundingBox:
    """Where an element sits on its page, in the parser's page coordinates."""

    left: float
    top: float
    right: float
    bottom: float


@dataclass(frozen=True)
class DocumentInfo:
    """One source document and the metadata every one of its chunks will carry."""

    doc_id: str
    """Stable identifier for the document."""
    doc_title: str
    """Title as it appears in citations."""
    source_path: Path
    sha256: str
    """Hash of the file contents. An unchanged hash means the file is skipped."""
    version: str | None
    effective_date: date | None
    access_group: str


@dataclass(frozen=True)
class Element:
    """One piece of a parsed document: a block of text, a table or a figure."""

    doc_id: str
    element_id: str
    page: int
    """Page number, counted from 1."""
    bbox: BoundingBox | None
    element_type: ElementType
    text: str
    """Text content. For a table, a text version that includes its title and column
    headers. For a figure, its caption (empty if it has none)."""
    heading_path: tuple[str, ...] = ()
    """Headings above this element, outermost first. A heading's own text is not in
    its own path."""
    html: str | None = None
    """Tables only: the original table as HTML."""
    image_path: Path | None = None
    """Figures only: the cropped image saved under ``paths.figures_dir``."""
    role: TextRole = TextRole.PARAGRAPH
    """Text elements only: heading, paragraph, list item and so on."""
    heading_level: int | None = None
    """Headings only: 1 for a top-level heading, 2 for the level below, and so on."""
    list_id: str | None = None
    """List items only: the same value for every item of one list, so that chunking can
    keep the list whole."""
    page_marks: tuple[PageMark, ...] = ()
    """Where ``text`` crosses onto a later page, if it does. Empty for an element that
    sits on one page."""


@dataclass(frozen=True)
class FlowchartEdge:
    """One arrow in a flowchart: ``source -> target``, with its condition if it has one."""

    source: str
    target: str
    condition: str | None = None


@dataclass(frozen=True)
class FigureDescription:
    """What the vision model wrote about one figure. Model-generated, so not authoritative."""

    image_sha256: str
    """Hash of the cropped image. The cache key: an unchanged figure is never re-described."""
    model: str
    """Vision model that wrote the description."""
    figure_type: str
    """For example bar chart, line chart, flowchart."""
    title: str | None
    axes_and_legend: str | None
    readable_values: tuple[str, ...] = ()
    """Values that are clearly readable in the figure, quoted as shown."""
    nodes: tuple[str, ...] = ()
    """Flowcharts only: the boxes."""
    edges: tuple[FlowchartEdge, ...] = ()
    """Flowcharts only: the arrows."""
    raw_text: str | None = None
    """The model's reply as it wrote it. Set only when the reply could not be read as
    the structured description that was asked for; the other fields are then empty."""


@dataclass(frozen=True)
class Chunk:
    """The unit that is embedded, searched and shown to the answer model."""

    chunk_id: str
    doc_id: str
    doc_title: str
    version: str | None
    effective_date: date | None
    page: int
    """Page the chunk starts on, counted from 1. This is the page that gets cited."""
    element_type: ElementType
    access_group: str
    text: str
    """The searchable text. For a figure, the vision model's description."""
    heading_path: tuple[str, ...] = ()
    html: str | None = None
    """Tables only: the original table as HTML, for display."""
    image_path: Path | None = None
    """Figures only: the cropped image, for display."""
    model_generated: bool = False
    """True when ``text`` is a figure description. The answer must then say so and tell
    the user to check the page."""
    page_end: int | None = None
    """Last page the chunk covers. ``None`` when it is all on ``page``."""
    page_marks: tuple[PageMark, ...] = ()
    """Where ``text`` crosses onto a later page. The answer model is shown these
    positions, so that it can cite the page a fact is actually on."""

    @property
    def last_page(self) -> int:
        """The last page the chunk covers (``page`` itself for a one-page chunk)."""
        return self.page if self.page_end is None else self.page_end

    def covers_page(self, page: int) -> bool:
        """True if ``page`` is one of the pages this chunk's text comes from."""
        return self.page <= page <= self.last_page


@dataclass(frozen=True)
class ScoredChunk:
    """A chunk that retrieval returned, with the scores that got it there."""

    chunk: Chunk
    fusion_score: float
    """Reciprocal rank fusion score from the dense and sparse searches."""
    rerank_score: float | None = None
    """Reranker score. ``None`` until the candidate has been reranked."""


@dataclass(frozen=True)
class RetrievalTimings:
    """Where the time of one search went, in seconds."""

    embed_s: float = 0.0
    """Turning the question into vectors."""
    search_s: float = 0.0
    """The dense and the sparse search in the vector store, and merging them."""
    rerank_s: float = 0.0
    """Scoring the candidates with the reranker. Usually the largest part."""
    candidates: int = 0
    """How many passages were reranked."""
    candidates_found: int = 0
    """How many passages hybrid search returned. More than ``candidates`` when
    ``retrieval.rerank_time_limit_s`` ended the reranking early."""


@dataclass(frozen=True)
class AnswerTimings:
    """Where the time of one answer went. Stored with every logged question."""

    retrieval: RetrievalTimings = RetrievalTimings()
    first_token_s: float | None = None
    """Seconds from sending the request to the answer model until its first piece of
    text. On a CPU this is mostly the model reading the evidence. ``None`` when the
    model was not called or the answer was not streamed."""
    generation_s: float = 0.0
    """Seconds the whole call to the answer model took."""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    evidence_chars: int = 0
    """Characters of evidence that were sent to the answer model, with the headers of
    its passages. The model's reading speed is measured in them."""
    time_limit_s: float | None = None
    """The time limit this answer was planned for (``generation.time_limit_s``)."""
    stopped_at_limit: bool = False
    """True if the writing was stopped because that limit had run out."""


@dataclass(frozen=True)
class RetrievalResult:
    """What retrieval found for one question."""

    question: str
    chunks: tuple[ScoredChunk, ...]
    """The top chunks after reranking, best first."""
    found: bool
    """False when the best rerank score is below the configured threshold. The reply is
    then the not-found message and the answer model is not called."""
    best_score: float | None = None
    """Best rerank score among the candidates. ``None`` when the search found nothing."""
    elapsed_s: float = 0.0
    """Seconds the search and reranking took. Added to the answer's latency."""
    timings: RetrievalTimings = RetrievalTimings()


@dataclass(frozen=True)
class Citation:
    """One source reference in an answer, shown as ``[document title, p. N]``."""

    doc_title: str
    page: int
    chunk_id: str


@dataclass(frozen=True)
class Answer:
    """The reply to one question, with everything needed to display and log it."""

    question: str
    text: str
    citations: tuple[Citation, ...]
    sources: tuple[ScoredChunk, ...]
    """The chunks the answer was written from, for the expandable sources in the UI."""
    refused: bool
    """True when the reply is the not-found message."""
    answer_model: str | None
    """Model that wrote the answer. ``None`` when refused, because no model was called."""
    latency_s: float
    unmatched_citations: tuple[str, ...] = ()
    """Citations in ``text`` that match none of ``sources``, exactly as written. A
    citation that cannot be traced to a retrieved chunk must not be trusted."""
    notes: tuple[str, ...] = ()
    """Sentences this program added after the model's reply, for example the warning
    that a figure description was used. They are already part of ``text``."""
    evidence_ids: tuple[str, ...] = ()
    """``chunk_id`` of every source that was actually given to the answer model. The
    weaker ones among ``sources`` are left out to keep the answer fast."""
    timings: AnswerTimings = AnswerTimings()


@dataclass(frozen=True)
class IngestReport:
    """Summary of one ingestion run."""

    files_seen: int
    files_skipped: int
    """Files whose hash had not changed since the last run."""
    files_ingested: int
    chunks_indexed: int
    failures: tuple[str, ...] = ()
    """One message per file that could not be processed."""
    warnings: tuple[str, ...] = ()
    """Things that did not stop a file but that a person should know about, for example
    a table too long to be embedded whole."""
    figures_described: int = 0
    """Figures the vision model was run on in this run. Cached descriptions do not count."""
    read_s: float = 0.0
    """Seconds spent reading files: parsing, or loading the cached parse."""
    figures_s: float = 0.0
    """Seconds spent describing figures."""
    index_s: float = 0.0
    """Seconds spent chunking, embedding and storing."""
    files_to_ingest: int = 0
    """Files this run set out to ingest: new or changed, after ``limit``."""
    stopped: bool = False
    """True if the run was asked to stop before it had been through all of them."""
