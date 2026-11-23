"""Turning one document's elements into chunks.

Tables and figures are simple: each becomes exactly one chunk and is never split.

Running text is cut by the document's own structure. The rules, in order of priority:

1. A chunk never exceeds ``chunking.max_tokens``.
2. A heading stays with the text under it. A chunk never ends with a heading.
3. A list stays whole. It is cut, between items, only if it is too long for any chunk.
4. A new section starts a new chunk once the current chunk has reached
   ``chunking.min_tokens``. Sections shorter than that are gathered into one chunk.
5. When the next paragraph does not fit: if the chunk has reached ``min_tokens`` it
   ends at the paragraph boundary; if not, the paragraph is cut at a sentence boundary
   and the chunk is filled up.
6. Where a section's text is cut (rule 5, or a block longer than a chunk), the next
   chunk starts by repeating the end of the previous one: whole sentences, between
   ``chunking.overlap_min_tokens`` and ``chunking.overlap_max_tokens`` tokens. A
   sentence that falls on the boundary is then complete in at least one chunk. There
   is no overlap across a heading.

A chunk can end up shorter than ``min_tokens`` in three cases: at the end of a
document, in front of a list that would not fit, and when a document is that short.

Every chunk records the page it starts on, and where inside its text a later page
begins (``page_marks``), so that an answer can cite the page a fact is actually on.

Token counts are approximate by a few tokens: the pieces of a chunk are counted
separately and added up. With BGE-M3's tokenizer the sum errs on the high side, which
is the safe direction for rule 1.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from docqa.config import ChunkingConfig, Config
from docqa.ingest.figures import description_text, is_informative
from docqa.schema import (
    Chunk,
    DocumentInfo,
    Element,
    ElementType,
    FigureDescription,
    PageMark,
    TextRole,
)
from docqa.tokens import TokenCounter

_BLOCK_GAP = "\n\n"
_GAP_TOKENS = 1  # allowance for the blank line between two blocks

# Words that end in a full stop without ending a sentence.
_ABBREVIATIONS = frozenset(
    "rs no nos sr dr mr mrs ms viz etc vs e.g i.e para paras sec cl ltd co pvt fig st "
    "art reg approx incl excl ref cf".split()
)
_SENTENCE_END = re.compile(r"[.!?][\"')\]]*\s+|\n+")
_WORD = re.compile(r"\S+\s*")


@dataclass(frozen=True)
class _Block:
    """A piece of text that packing treats as one unit: a heading, a paragraph or a list."""

    kind: str
    """``heading``, ``body`` or ``overlap``."""
    text: str
    tokens: int
    page: int
    """Page the block starts on."""
    marks: tuple[PageMark, ...]
    """Page changes inside ``text``."""
    heading_path: tuple[str, ...]
    order: int
    """Index of the block's first element in the document, to keep chunks in reading order."""
    units: tuple[tuple[int, int], ...] = ()
    """Lists only: the character span of each item. A list may be cut only between items."""

    @property
    def is_list(self) -> bool:
        return bool(self.units)

    @property
    def last_page(self) -> int:
        return self.marks[-1][1] if self.marks else self.page

    def page_at(self, offset: int) -> int:
        page = self.page
        for mark_offset, mark_page in self.marks:
            if mark_offset <= offset:
                page = mark_page
        return page


def chunk_elements(
    elements: Sequence[Element],
    info: DocumentInfo,
    descriptions: Mapping[str, FigureDescription],
    config: Config,
    *,
    counter: TokenCounter | None = None,
) -> list[Chunk]:
    """Turn one document's elements into chunks.

    Text is chunked by heading structure into ``chunking.min_tokens`` to
    ``chunking.max_tokens`` tokens, with an overlap between
    ``chunking.overlap_min_tokens`` and ``chunking.overlap_max_tokens``. A heading
    stays with its paragraphs and a list stays whole.

    A table is always exactly one chunk; it is never split. A figure is one chunk whose
    text is its description, marked as model-generated. ``descriptions`` maps
    ``element_id`` to the description of that figure. A figure without a description
    becomes a chunk only if it has a caption, and that chunk holds the caption alone.

    Every chunk carries ``doc_id``, ``doc_title``, ``version``, ``effective_date``,
    ``page``, ``element_type`` and ``access_group`` from ``info``.

    Args:
        counter: What counts tokens. Defaults to the embedding model's own tokenizer,
            which needs that model's files on this machine.
    """
    if counter is None:
        from docqa import resources

        counter = resources.get_token_counter(config)

    drafts: list[tuple[int, dict]] = []  # (reading order, fields of the chunk)

    for index, element in enumerate(elements):
        if element.element_type is ElementType.TABLE:
            pages = [element.page, *(page for _offset, page in element.page_marks)]
            drafts.append(
                (
                    index,
                    {
                        "element_type": ElementType.TABLE,
                        "text": element.text,
                        "page": element.page,
                        "page_end": max(pages) if max(pages) > element.page else None,
                        "heading_path": element.heading_path,
                        "html": element.html,
                    },
                )
            )
        elif element.element_type is ElementType.FIGURE:
            description = descriptions.get(element.element_id)
            if description is not None:
                if not is_informative(description):
                    continue
                text, generated = description_text(description, element.text), True
            elif element.text.strip():
                text, generated = f"Figure caption: {element.text.strip()}", False
            else:
                continue
            drafts.append(
                (
                    index,
                    {
                        "element_type": ElementType.FIGURE,
                        "text": text,
                        "page": element.page,
                        "heading_path": element.heading_path,
                        "image_path": element.image_path,
                        "model_generated": generated,
                    },
                )
            )

    blocks = _blocks(elements, counter)
    for group in _pack(blocks, config.chunking, counter):
        text, page, marks = _assemble(group)
        last_page = max([page, *(mark_page for _offset, mark_page in marks)])
        first_own = next((block for block in group if block.kind != "overlap"), group[0])
        drafts.append(
            (
                first_own.order,
                {
                    "element_type": ElementType.TEXT,
                    "text": text,
                    "page": page,
                    "page_end": last_page if last_page > page else None,
                    "page_marks": marks,
                    "heading_path": group[0].heading_path,
                },
            )
        )

    drafts.sort(key=lambda draft: draft[0])
    return [
        Chunk(
            chunk_id=f"{info.doc_id}:{number:05d}",
            doc_id=info.doc_id,
            doc_title=info.doc_title,
            version=info.version,
            effective_date=info.effective_date,
            access_group=info.access_group,
            **fields,
        )
        for number, (_order, fields) in enumerate(drafts)
    ]


# -- from elements to blocks ----------------------------------------------------------


def _blocks(elements: Sequence[Element], counter: TokenCounter) -> list[_Block]:
    """Group the text elements into headings, paragraphs and whole lists."""
    blocks: list[_Block] = []
    items: list[tuple[int, Element]] = []  # the list being collected

    def end_list() -> None:
        if not items:
            return
        parts: list[str] = []
        marks: list[PageMark] = []
        units: list[tuple[int, int]] = []
        offset = 0
        page = items[0][1].page
        for position, (_index, item) in enumerate(items):
            if position:
                offset += 1  # the line break between two items
            if item.page != page:
                marks.append((offset, item.page))
                page = item.page
            for mark_offset, mark_page in item.page_marks:
                marks.append((offset + mark_offset, mark_page))
                page = mark_page
            units.append((offset, offset + len(item.text)))
            parts.append(item.text)
            offset += len(item.text)
        text = "\n".join(parts)
        first_index, first = items[0]
        blocks.append(
            _Block(
                kind="body",
                text=text,
                tokens=counter.count(text),
                page=first.page,
                marks=tuple(marks),
                heading_path=first.heading_path,
                order=first_index,
                units=tuple(units),
            )
        )
        items.clear()

    for index, element in enumerate(elements):
        if element.element_type is not ElementType.TEXT:
            end_list()  # a table or figure between two items ends the list
            continue
        if not element.text.strip():
            continue
        if element.role is TextRole.LIST_ITEM:
            if items and (
                items[0][1].list_id != element.list_id
                or items[0][1].heading_path != element.heading_path
            ):
                end_list()
            items.append((index, element))
            continue
        end_list()
        kind = "heading" if element.role in (TextRole.TITLE, TextRole.HEADING) else "body"
        blocks.append(
            _Block(
                kind=kind,
                text=element.text,
                tokens=counter.count(element.text),
                page=element.page,
                marks=element.page_marks,
                heading_path=element.heading_path,
                order=index,
            )
        )
    end_list()
    return blocks


# -- packing blocks into chunks -------------------------------------------------------


def _pack(blocks: Sequence[_Block], sizes: ChunkingConfig, counter: TokenCounter) -> list[list[_Block]]:
    """Decide which blocks go into which chunk. See the rules at the top of this file."""
    max_tokens, min_tokens = sizes.max_tokens, sizes.min_tokens
    # A piece cut from a long block must leave room for the overlap that precedes it.
    piece_limit = max(max_tokens - sizes.overlap_max_tokens - _GAP_TOKENS, 1)

    chunks: list[list[_Block]] = []
    current: list[_Block] = []
    size = 0

    def add(block: _Block) -> None:
        nonlocal size
        size += block.tokens + (_GAP_TOKENS if current else 0)
        current.append(block)

    def close() -> None:
        """End the current chunk. Headings at its end move on into the next one (rule 2)."""
        nonlocal current, size
        carried: list[_Block] = []
        while current and current[-1].kind == "heading":
            carried.insert(0, current.pop())
        if current:
            chunks.append(current)
        current, size = [], 0
        for block in carried:
            add(block)

    def continue_in_new_chunk(next_tokens: int) -> None:
        """Close the chunk and start the next with the overlap, if the text runs on (rule 6)."""
        close()
        if current or not chunks:
            return  # a heading came in between, or there is nothing to repeat
        overlap = _overlap(chunks[-1], sizes, counter)
        if overlap is not None and overlap.tokens + _GAP_TOKENS + next_tokens <= max_tokens:
            add(overlap)

    def fits(block: _Block) -> bool:
        return size + block.tokens + (_GAP_TOKENS if current else 0) <= max_tokens

    for block in blocks:
        if block.kind == "heading":
            if size >= min_tokens or not fits(block):
                close()  # rule 4
            add(block)
            continue

        if fits(block):
            add(block)
            continue

        has_text = any(item.kind != "heading" for item in current)
        list_fits_alone = block.is_list and block.tokens <= max_tokens
        if has_text and (size >= min_tokens or list_fits_alone):
            continue_in_new_chunk(block.tokens)  # rules 3 and 5
            if fits(block):
                add(block)
                continue

        # The block has to be cut: it tops up a short chunk, or it is too long for any chunk.
        has_text = any(item.kind != "heading" for item in current)
        room = max_tokens - size - (_GAP_TOKENS if current else 0)
        if not has_text and room < 1:
            # Headings that fill a whole chunk by themselves: nothing can be kept with them.
            chunks.append(current)
            current, size = [], 0
            room = max_tokens
        # Under headings alone, every piece must fit in the room the headings leave.
        limit = piece_limit if has_text else min(piece_limit, room)
        pieces = _split(block, room, limit, counter)
        fresh = False  # True once a new chunk has been started for the piece that comes next
        for number, piece in enumerate(pieces):
            if piece is None:  # not even one sentence fits in what is left of this chunk
                following = pieces[number + 1]
                continue_in_new_chunk(following.tokens if following is not None else 0)
                fresh = True
                continue
            if number > 0 and not fresh:
                continue_in_new_chunk(piece.tokens)
            fresh = False
            add(piece)

    close()
    if current:  # headings at the very end of the document, with nothing under them
        if chunks and sum(b.tokens for b in chunks[-1]) + size + _GAP_TOKENS <= max_tokens:
            chunks[-1].extend(current)
        else:
            chunks.append(current)

    # A short last chunk joins the one before it, if the two fit together.
    if len(chunks) >= 2:
        last, before = chunks[-1], chunks[-2]
        own = [block for block in last if block.kind != "overlap"]
        own_tokens = sum(block.tokens for block in own) + _GAP_TOKENS * len(own)
        before_tokens = sum(block.tokens for block in before) + _GAP_TOKENS * (len(before) - 1)
        if own_tokens < min_tokens and before_tokens + own_tokens <= max_tokens:
            before.extend(own)
            chunks.pop()
    return chunks


def _split(
    block: _Block, first_limit: int, limit: int, counter: TokenCounter
) -> list[_Block | None]:
    """Cut a block into pieces. The first holds at most ``first_limit`` tokens, the rest ``limit``.

    A list is cut between items, a paragraph between sentences, and a sentence longer
    than a whole chunk between words. The first piece is ``None`` when ``first_limit``
    has no room for even the first sentence.
    """
    atoms: list[tuple[int, int, int]] = []  # (start, end, tokens); each fits in ``limit``
    for start, end in block.units or sentence_spans(block.text):
        atoms.extend(_fit(block.text, start, end, limit, counter, by_sentence=block.is_list))

    groups: list[list[tuple[int, int, int]]] = []
    group: list[tuple[int, int, int]] = []
    used, cap = 0, first_limit
    for atom in atoms:
        if group and used + atom[2] > cap:
            groups.append(group)
            group, used, cap = [], 0, limit
        elif not group and not groups and atom[2] > cap:
            groups.append([])  # nothing fits in the first piece
            cap = limit
        group.append(atom)
        used += atom[2]
    if group:
        groups.append(group)

    pieces: list[_Block | None] = []
    for group in groups:
        if not group:
            pieces.append(None)
            continue
        pieces.append(_slice(block, group[0][0], group[-1][1], sum(atom[2] for atom in group)))
    return pieces


def _fit(
    text: str, start: int, end: int, limit: int, counter: TokenCounter, *, by_sentence: bool
) -> list[tuple[int, int, int]]:
    """Spans inside ``text[start:end]`` that each hold at most ``limit`` tokens."""
    tokens = counter.count(text[start:end])
    if tokens <= limit:
        return [(start, end, tokens)]
    if by_sentence:
        spans = [(start + s, start + e) for s, e in sentence_spans(text[start:end])]
        if len(spans) > 1:
            return [
                atom
                for s, e in spans
                for atom in _fit(text, s, e, limit, counter, by_sentence=False)
            ]
    # One sentence longer than a whole chunk: cut between words.
    atoms: list[tuple[int, int, int]] = []
    piece_start, used = start, 0
    for match in _WORD.finditer(text, start, end):
        word_tokens = max(counter.count(match.group()), 1)
        if used and used + word_tokens > limit:
            atoms.append((piece_start, match.start(), used))
            piece_start, used = match.start(), 0
        used += word_tokens
    if used:
        atoms.append((piece_start, end, used))
    return atoms


def _slice(block: _Block, start: int, end: int, tokens: int) -> _Block:
    """The part of a block between two character offsets, with its page and page marks."""
    text = block.text[start:end]
    start += len(text) - len(text.lstrip())
    text = text.strip()
    end = start + len(text)
    return _Block(
        kind="body",
        text=text,
        tokens=tokens,
        page=block.page_at(start),
        marks=tuple((offset - start, page) for offset, page in block.marks if start < offset < end),
        heading_path=block.heading_path,
        order=block.order,
    )


def _overlap(previous: Sequence[_Block], sizes: ChunkingConfig, counter: TokenCounter) -> _Block | None:
    """The end of the previous chunk, to be repeated at the start of the next (rule 6)."""
    if sizes.overlap_max_tokens <= 0:
        return None
    # Take text blocks from the end until there is enough to choose from. Stop at a heading.
    tail: list[_Block] = []
    gathered = 0
    for block in reversed(previous):
        if block.kind == "heading":
            break
        tail.insert(0, block)
        gathered += block.tokens
        if gathered >= sizes.overlap_max_tokens:
            break
    if not tail:
        return None
    text, page, marks = _assemble(tail)
    source = _Block("body", text, gathered, page, marks, tail[-1].heading_path, tail[-1].order)

    # Whole sentences from the end, as many as fit in overlap_max.
    start, used = len(text), 0
    for span_start, span_end in reversed(sentence_spans(text)):
        tokens = counter.count(text[span_start:span_end])
        if used + tokens > sizes.overlap_max_tokens:
            if used < sizes.overlap_min_tokens:
                # The next sentence is too long to take whole: take its last words.
                for match in reversed(list(_WORD.finditer(text, span_start, span_end))):
                    word_tokens = max(counter.count(match.group()), 1)
                    if used + word_tokens > sizes.overlap_max_tokens:
                        break
                    start, used = match.start(), used + word_tokens
            break
        start, used = span_start, used + tokens
    if used == 0:
        return None
    piece = _slice(source, start, len(text), used)
    return _Block("overlap", piece.text, piece.tokens, piece.page, piece.marks, piece.heading_path, piece.order)


def _assemble(blocks: Sequence[_Block]) -> tuple[str, int, tuple[PageMark, ...]]:
    """Join blocks into one text. Returns the text, its first page, and its page marks."""
    parts: list[str] = []
    marks: list[PageMark] = []
    offset = 0
    page = blocks[0].page
    for position, block in enumerate(blocks):
        if position:
            offset += len(_BLOCK_GAP)
        if block.page != page:
            marks.append((offset, block.page))
            page = block.page
        for mark_offset, mark_page in block.marks:
            if mark_page != page:
                marks.append((offset + mark_offset, mark_page))
                page = mark_page
        parts.append(block.text)
        offset += len(block.text)
    return _BLOCK_GAP.join(parts), blocks[0].page, tuple(marks)


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of the sentences in ``text``. Together they cover all of it.

    A sentence ends at a full stop, question mark or exclamation mark that is followed
    by white space and then by something that can start a sentence, or at a line
    break. A full stop after an abbreviation such as "Rs." or "No.", or after a single
    letter, does not end a sentence. This is a splitter for choosing where to cut a
    chunk, not a linguistic one: an occasional wrong cut costs little.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        end = match.end()
        if end >= len(text):
            break
        if "\n" not in match.group():
            before = text[start : match.start()].rsplit(None, 1)
            last_word = before[-1].lower().strip("\"'([") if before else ""
            if text[match.start()] == "." and (last_word in _ABBREVIATIONS or len(last_word) == 1):
                continue
            following = text[end]
            if not (following.isupper() or following.isdigit() or following in "\"'(["):
                continue
        spans.append((start, end))
        start = end
    if start < len(text):
        spans.append((start, len(text)))
    return spans
