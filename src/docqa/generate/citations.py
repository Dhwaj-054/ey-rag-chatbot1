"""Reading the citations out of an answer and tying each one to a retrieved chunk.

The answer model is told to cite as ``[document title, p. N]``. A citation only means
something if it points at a passage the model was actually given. So every citation
found in an answer is matched against the retrieved chunks:

- the title has to be the title of a retrieved document (compared without regard to
  case, spacing or the kind of dash; a title the model shortened is accepted if it
  fits exactly one retrieved document);
- the page has to be one of the pages that a retrieved chunk of that document covers.

A citation that passes becomes a ``Citation`` that names the chunk. One that does not
is returned as written, so that the reader can be told it could not be checked.

A small model sometimes cites the evidence by its number instead of copying a long
title: ``[Evidence 2, p. 14]`` or ``[2, p. 14]``. That is not the form it was asked
for, but it is unambiguous, so it is accepted and resolved to the document that
evidence 2 came from.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from docqa.schema import Citation, ScoredChunk

# [title, p. 12]   [title, p.12]   [title, pp. 12-13]   [title, page 12]   [title, pages 12 to 13]
CITATION = re.compile(
    r"\[\s*(?P<title>[^\[\]]{1,400}?)\s*,\s*(?:pp?\.?|pages?)\s*(?P<first>\d{1,5})"
    r"(?:\s*(?:-|–|—|to|and|&|,)\s*(?:pp?\.?\s*)?(?P<last>\d{1,5}))?\s*\]",
    re.IGNORECASE,
)
_EVIDENCE_NUMBER = re.compile(r"^(?:evidence\s*)?(\d{1,2})$")
_MIN_PARTIAL_TITLE = 8
_MAX_PAGE_RANGE = 20


def normalise_title(title: str) -> str:
    """A title reduced to what has to match: lower case, single spaces, plain dashes and quotes."""
    title = title.replace("–", "-").replace("—", "-").replace("’", "'").replace("“", '"').replace("”", '"')
    title = re.sub(r"\s+", " ", title).strip().strip("\"'.,;:").strip()
    return title.casefold()


def match_citations(
    text: str, sources: Sequence[ScoredChunk]
) -> tuple[tuple[Citation, ...], tuple[str, ...]]:
    """Find the citations in ``text`` and match them to ``sources``.

    Returns ``(matched, unmatched)``. ``matched`` holds one ``Citation`` per distinct
    document and page, in the order they first appear, each naming the best-ranked
    chunk that covers that page. ``unmatched`` holds, as written, every citation that
    names a document or a page that was not among the sources.
    """
    titles: dict[str, list[ScoredChunk]] = {}
    for source in sources:
        titles.setdefault(normalise_title(source.chunk.doc_title), []).append(source)

    matched: dict[tuple[str, int], Citation] = {}
    unmatched: dict[str, None] = {}
    for found in CITATION.finditer(text):
        first = int(found["first"])
        last = int(found["last"]) if found["last"] else first
        if last < first or last - first > _MAX_PAGE_RANGE:
            last = first
        cited_title = normalise_title(found["title"])
        by_number = _EVIDENCE_NUMBER.match(cited_title)
        if by_number and 1 <= int(by_number.group(1)) <= len(sources):
            doc_id = sources[int(by_number.group(1)) - 1].chunk.doc_id
            candidates = [source for source in sources if source.chunk.doc_id == doc_id]
        else:
            candidates = _chunks_for_title(cited_title, titles)
        complete = bool(candidates)
        for page in range(first, last + 1):
            chunk = next((s.chunk for s in candidates if s.chunk.covers_page(page)), None)
            if chunk is None:
                complete = False
                continue
            matched.setdefault(
                (chunk.doc_id, page), Citation(doc_title=chunk.doc_title, page=page, chunk_id=chunk.chunk_id)
            )
        if not complete:
            unmatched.setdefault(found.group(0))
    return tuple(matched.values()), tuple(unmatched)


def _chunks_for_title(wanted: str, titles: dict[str, list[ScoredChunk]]) -> list[ScoredChunk]:
    """The retrieved chunks of the document a cited title refers to, best first."""
    if wanted in titles:
        return titles[wanted]
    if len(wanted) < _MIN_PARTIAL_TITLE:
        return []
    partial = [title for title in titles if wanted in title or title in wanted]
    doc_ids = {source.chunk.doc_id for title in partial for source in titles[title]}
    if len(doc_ids) != 1:
        return []  # no document fits, or more than one does: do not guess
    return [source for title in partial for source in titles[title]]
