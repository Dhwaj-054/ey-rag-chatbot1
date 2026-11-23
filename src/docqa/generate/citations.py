"""Reading the citations out of an answer and tying each one to a passage.

The answer model is told to cite by evidence number and page: ``[E1, p. 12]``. This
module checks every such reference against the passages the model was given, and
rewrites the ones that hold into the form the reader sees: ``[document title, p. 12]``.

A reference holds if

- its label is the number of a passage the model was given (``E1``, ``Evidence 1`` and
  a bare ``1`` are all read as passage 1), or it is the title of a document among the
  passages (compared without regard to case, spacing or the kind of dash; a title the
  model shortened is accepted if it fits exactly one of the documents);
- and its page is one of the pages that a passage of that document covers.

A reference that holds becomes a ``Citation`` that names the passage. One that does
not is left in the text as the model wrote it and is reported, so that the reader can
be told it could not be checked.

A reference without a page, ``[E1]``, is accepted too. It is rewritten with the page,
or the page range, of that passage: the model did not say which page, so the citation
must not claim more than the passage does.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from docqa.schema import Chunk, Citation, ScoredChunk

# [E1, p. 12]  [title, p. 12]  [title, p.12]  [title, pp. 12-13]  [title, page 12]  [E1]
CITATION = re.compile(
    r"\[\s*(?P<title>[^\[\]]{1,400}?)\s*,\s*(?:pp?\.?|pages?)\s*(?P<first>\d{1,5})"
    r"(?:\s*(?:-|–|—|to|and|&|,)\s*(?:pp?\.?\s*)?(?P<last>\d{1,5}))?\s*\]"
    r"|\[\s*(?P<bare>E\s?\d{1,2})\s*\]",
    re.IGNORECASE,
)
_EVIDENCE_LABEL = re.compile(r"^(?:evidence\s*|e\s?)?(\d{1,2})$")
_MIN_PARTIAL_TITLE = 8
_MAX_PAGE_RANGE = 20


def normalise_title(title: str) -> str:
    """A title reduced to what has to match: lower case, single spaces, plain dashes and quotes."""
    title = title.replace("–", "-").replace("—", "-").replace("’", "'").replace("“", '"').replace("”", '"')
    title = re.sub(r"\s+", " ", title).strip().strip("\"'.,;:").strip()
    return title.casefold()


def format_citation(doc_title: str, first: int, last: int | None = None) -> str:
    """A citation as the reader sees it: ``[document title, p. N]``."""
    if last is not None and last != first:
        return f"[{doc_title}, pp. {first}-{last}]"
    return f"[{doc_title}, p. {first}]"


def rewrite_citations(
    text: str, sources: Sequence[ScoredChunk]
) -> tuple[str, tuple[Citation, ...], tuple[str, ...]]:
    """Check the references in ``text`` and write the valid ones out in full.

    ``sources`` are the passages the model was given, best first: passage 1 is ``E1``.

    Returns ``(text, citations, unmatched)``. In the returned text every reference that
    holds reads ``[document title, p. N]``. ``citations`` has one entry per distinct
    document and page, in the order they first appear, each naming the best-ranked
    passage that covers that page. ``unmatched`` holds, as written, every reference to
    a passage, document or page that the model was not given.
    """
    titles: dict[str, list[ScoredChunk]] = {}
    for source in sources:
        titles.setdefault(normalise_title(source.chunk.doc_title), []).append(source)

    matched: dict[tuple[str, int], Citation] = {}
    unmatched: dict[str, None] = {}

    def cite(chunk: Chunk, page: int) -> None:
        matched.setdefault(
            (chunk.doc_id, page), Citation(doc_title=chunk.doc_title, page=page, chunk_id=chunk.chunk_id)
        )

    def replace(found: re.Match[str]) -> str:
        if found["bare"]:
            number = int(re.sub(r"\D", "", found["bare"]))
            if not 1 <= number <= len(sources):
                unmatched.setdefault(found.group(0))
                return found.group(0)
            chunk = sources[number - 1].chunk
            cite(chunk, chunk.page)
            return format_citation(chunk.doc_title, chunk.page, chunk.last_page)

        first = int(found["first"])
        last = int(found["last"]) if found["last"] else first
        if last < first or last - first > _MAX_PAGE_RANGE:
            last = first
        candidates = _passages_for(normalise_title(found["title"]), sources, titles)
        covering = [
            next((s.chunk for s in candidates if s.chunk.covers_page(page)), None)
            for page in range(first, last + 1)
        ]
        for page, chunk in zip(range(first, last + 1), covering, strict=True):
            if chunk is not None:
                cite(chunk, page)
        if not candidates or any(chunk is None for chunk in covering):
            unmatched.setdefault(found.group(0))
            return found.group(0)
        return format_citation(covering[0].doc_title, first, last)

    rewritten = CITATION.sub(replace, text)
    return rewritten, tuple(matched.values()), tuple(unmatched)


def match_citations(
    text: str, sources: Sequence[ScoredChunk]
) -> tuple[tuple[Citation, ...], tuple[str, ...]]:
    """As ``rewrite_citations``, without the rewritten text: ``(citations, unmatched)``."""
    _text, citations, unmatched = rewrite_citations(text, sources)
    return citations, unmatched


def _passages_for(
    label: str, sources: Sequence[ScoredChunk], titles: dict[str, list[ScoredChunk]]
) -> list[ScoredChunk]:
    """The passages of the document that a reference's label points at, best first."""
    by_number = _EVIDENCE_LABEL.match(label)
    if by_number:
        number = int(by_number.group(1))
        if not 1 <= number <= len(sources):
            return []
        doc_id = sources[number - 1].chunk.doc_id
        return [source for source in sources if source.chunk.doc_id == doc_id]
    if label in titles:
        return titles[label]
    if len(label) < _MIN_PARTIAL_TITLE:
        return []
    partial = [title for title in titles if label in title or title in label]
    doc_ids = {source.chunk.doc_id for title in partial for source in titles[title]}
    if len(doc_ids) != 1:
        return []  # no document fits, or more than one does: do not guess
    return [source for title in partial for source in titles[title]]
