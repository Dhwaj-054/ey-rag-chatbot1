"""Chunking. Tokens are counted as words here, so every size in a test can be read off.

The sizes are scaled down from the project's 400 to 800 tokens: a chunk holds 40 to 80
words and the overlap is 5 to 10 words.
"""

from __future__ import annotations

import random
from datetime import date
from pathlib import Path

import pytest

from docqa.ingest.chunker import chunk_elements, sentence_spans
from docqa.schema import (
    DocumentInfo,
    Element,
    ElementType,
    FigureDescription,
    FlowchartEdge,
    TextRole,
)
from docqa.tokens import TokenCounterError, WordCounter

MIN, MAX, OVERLAP_MIN, OVERLAP_MAX = 40, 80, 5, 10
COUNT = WordCounter()

INFO = DocumentInfo(
    doc_id="doc1",
    doc_title="Lending Policy",
    source_path=Path("Lending Policy.pdf"),
    sha256="0" * 64,
    version="v2",
    effective_date=date(2026, 4, 1),
    access_group="retail",
)


@pytest.fixture
def cfg(temp_config):
    def small_chunks(raw):
        raw["chunking"] = {
            "min_tokens": MIN,
            "max_tokens": MAX,
            "overlap_min_tokens": OVERLAP_MIN,
            "overlap_max_tokens": OVERLAP_MAX,
        }

    return temp_config(small_chunks)


def sentences(count: int, words_each: int, tag: str) -> str:
    """``count`` sentences of ``words_each`` words. Every word is unique and capitalised."""
    return " ".join(
        " ".join(f"{tag}{s}x{w}" for w in range(words_each - 1)) + f" {tag}{s}end."
        for s in range(count)
    )


class Doc:
    """Builds a list of elements the way the extraction step would."""

    def __init__(self) -> None:
        self.elements: list[Element] = []
        self._path: tuple[str, ...] = ()

    def _add(self, element_type: ElementType, text: str, page: int, **more) -> Element:
        element = Element(
            doc_id="doc1",
            element_id=f"doc1:e{len(self.elements):05d}",
            page=page,
            bbox=None,
            element_type=element_type,
            text=text,
            **more,
        )
        self.elements.append(element)
        return element

    def heading(self, text: str, page: int = 1) -> Doc:
        self._add(ElementType.TEXT, text, page, role=TextRole.HEADING, heading_level=1)
        self._path = (text,)
        return self

    def para(self, text: str, page: int = 1, marks=()) -> Doc:
        self._add(ElementType.TEXT, text, page, heading_path=self._path, page_marks=tuple(marks))
        return self

    def items(self, texts: list[str], page: int = 1, list_id: str = "#/groups/0") -> Doc:
        for text in texts:
            self._add(
                ElementType.TEXT, text, page, heading_path=self._path, role=TextRole.LIST_ITEM, list_id=list_id
            )
        return self

    def table(self, text: str, page: int = 1, html: str = "<table></table>") -> Element:
        return self._add(ElementType.TABLE, text, page, heading_path=self._path, html=html)

    def figure(self, caption: str = "", page: int = 1, image: str | None = "fig.png") -> Element:
        return self._add(
            ElementType.FIGURE,
            caption,
            page,
            heading_path=self._path,
            image_path=Path(image) if image else None,
        )


def chunk(doc: Doc, cfg, descriptions=None):
    return chunk_elements(doc.elements, INFO, descriptions or {}, cfg, counter=COUNT)


def size(chunk) -> int:
    return COUNT.count(chunk.text)


# -- structure ------------------------------------------------------------------------


def test_a_short_document_is_one_chunk_with_its_heading(cfg):
    body = sentences(2, 10, "A")
    doc = Doc().heading("1. Fees").para(body)

    chunks = chunk(doc, cfg)

    assert len(chunks) == 1
    assert chunks[0].text == f"1. Fees\n\n{body}"
    assert chunks[0].chunk_id == "doc1:00000"
    assert chunks[0].heading_path == ()
    assert chunks[0].element_type is ElementType.TEXT
    assert chunks[0].page == 1 and chunks[0].page_end is None and chunks[0].page_marks == ()


def test_every_chunk_carries_the_documents_metadata(cfg):
    only = chunk(Doc().para(sentences(2, 10, "A")), cfg)[0]

    assert (only.doc_id, only.doc_title, only.version) == ("doc1", "Lending Policy", "v2")
    assert only.effective_date == date(2026, 4, 1)
    assert only.access_group == "retail"
    assert only.model_generated is False


def test_short_sections_are_gathered_into_one_chunk(cfg):
    doc = Doc()
    for name in "ABC":
        doc.heading(f"Section {name}").para(sentences(1, 12, name))

    chunks = chunk(doc, cfg)

    assert len(chunks) == 1
    assert [line for line in chunks[0].text.split("\n\n") if line.startswith("Section")] == [
        "Section A",
        "Section B",
        "Section C",
    ]


def test_a_new_section_starts_a_new_chunk_once_the_minimum_is_reached(cfg):
    first, second = sentences(5, 10, "A"), sentences(5, 10, "B")
    doc = Doc().heading("Section A").para(first).heading("Section B").para(second)

    chunks = chunk(doc, cfg)

    assert [c.text for c in chunks] == [f"Section A\n\n{first}", f"Section B\n\n{second}"]
    # Nothing is repeated across a heading.
    assert not set(chunks[0].text.split()) & set(chunks[1].text.split()) - {"Section"}


def test_a_chunk_never_ends_with_a_heading(cfg):
    doc = (
        Doc()
        .heading("Section A")
        .para(sentences(5, 10, "A"))
        .heading("Section B")
        .heading("Sub-section B1")
        .para(sentences(6, 10, "B"))
    )

    chunks = chunk(doc, cfg)

    assert len(chunks) == 2
    assert chunks[0].text.split("\n\n")[-1].endswith("A4end.")
    assert chunks[1].text.startswith("Section B\n\nSub-section B1\n\n")


def test_a_short_chunk_is_filled_up_from_the_next_paragraph(cfg):
    """Rule 5: below the minimum, the next paragraph is cut at a sentence boundary."""
    doc = Doc().heading("Section A").para(sentences(3, 10, "A")).para(sentences(8, 10, "B"))

    first, second = chunk(doc, cfg)

    assert size(first) >= MIN
    assert first.text.rstrip().endswith("end.")  # cut between sentences, not inside one
    assert "B0x0" in first.text and "B7end." in second.text
    assert second.heading_path == ("Section A",)


def test_a_paragraph_that_does_not_fit_starts_the_next_chunk_when_the_minimum_is_reached(cfg):
    doc = Doc().para(sentences(5, 10, "A")).para(sentences(5, 10, "B"))

    first, second = chunk(doc, cfg)

    assert "B0x0" not in first.text
    assert second.text.endswith(sentences(5, 10, "B"))


# -- size and coverage ----------------------------------------------------------------


def test_no_text_chunk_exceeds_the_maximum_and_nothing_is_lost(cfg):
    randomness = random.Random(7)
    doc = Doc()
    for section in range(12):
        doc.heading(f"Heading{section} of section")
        for block in range(randomness.randint(1, 4)):
            tag = f"S{section}B{block}Q"
            if randomness.random() < 0.3:
                doc.items([sentences(1, randomness.randint(3, 12), f"{tag}I{i}") for i in range(randomness.randint(2, 9))])
            else:
                doc.para(sentences(randomness.randint(1, 14), randomness.randint(4, 25), tag))

    chunks = chunk(doc, cfg)

    assert all(size(c) <= MAX for c in chunks), [size(c) for c in chunks]
    produced = " ".join(c.text for c in chunks).split()
    for element in doc.elements:
        for word in element.text.split():
            assert word in produced, word
    # The first appearance of each source word is in reading order.
    source = [word for element in doc.elements for word in element.text.split()]
    first_seen = {word: position for position, word in reversed(list(enumerate(produced)))}
    positions = [first_seen[word] for word in dict.fromkeys(source)]
    assert positions == sorted(positions)


def test_a_sentence_longer_than_a_chunk_is_cut_between_words(cfg):
    endless = " ".join(f"W{n}" for n in range(200))

    chunks = chunk(Doc().para(endless), cfg)

    assert len(chunks) >= 3
    assert all(size(c) <= MAX for c in chunks)
    assert [w for w in dict.fromkeys(" ".join(c.text for c in chunks).split())] == endless.split()


# -- overlap --------------------------------------------------------------------------


def test_the_next_chunk_repeats_the_last_sentence_of_the_previous_one(cfg):
    doc = Doc().heading("Long section").para(sentences(30, 10, "L"))

    chunks = chunk(doc, cfg)

    assert len(chunks) >= 4
    for previous, following in zip(chunks, chunks[1:], strict=False):
        last_sentence = " ".join(previous.text.split()[-10:])  # every sentence has ten words
        assert following.text.startswith(last_sentence)
        assert OVERLAP_MIN <= len(last_sentence.split()) <= OVERLAP_MAX
        assert following.heading_path == ("Long section",)


def test_overlap_takes_the_last_words_when_a_whole_sentence_is_too_long(cfg):
    doc = Doc().para(sentences(8, 30, "L"))

    chunks = chunk(doc, cfg)

    for previous, following in zip(chunks, chunks[1:], strict=False):
        tail = " ".join(previous.text.split()[-OVERLAP_MAX:])
        assert following.text.startswith(tail)
        assert not following.text.startswith(" ".join(previous.text.split()[-(OVERLAP_MAX + 1) :]))


# -- lists ----------------------------------------------------------------------------


def test_a_list_is_not_cut_when_it_can_go_into_the_next_chunk(cfg):
    items = [sentences(1, 8, f"I{n}") for n in range(5)]
    doc = Doc().para(sentences(5, 10, "A")).items(items)

    first, second = chunk(doc, cfg)

    assert not any(item in first.text for item in items)
    assert "\n".join(items) in second.text


def test_a_list_longer_than_a_chunk_is_cut_only_between_items(cfg):
    items = [sentences(1, 6, f"I{n}") for n in range(30)]

    chunks = chunk(Doc().heading("Conditions").items(items), cfg)

    assert len(chunks) >= 3
    assert all(size(c) <= MAX for c in chunks)
    for item in items:
        assert any(item in c.text.split("\n") for c in chunks), item


def test_two_lists_in_a_row_are_two_blocks(cfg):
    doc = Doc().items(["- one", "- two"], list_id="#/groups/0").items(["- three"], list_id="#/groups/1")

    assert chunk(doc, cfg)[0].text == "- one\n- two\n\n- three"


# -- pages ----------------------------------------------------------------------------


def test_a_chunk_records_its_first_page_and_where_later_pages_begin(cfg):
    one, two, three = sentences(2, 10, "A"), sentences(2, 10, "B"), sentences(2, 10, "C")
    doc = Doc().para(one, page=4).para(two, page=4).para(three, page=5)

    only = chunk(doc, cfg)[0]

    assert (only.page, only.page_end) == (4, 5)
    assert only.page_marks == ((len(one) + 2 + len(two) + 2, 5),)
    assert only.text[only.page_marks[0][0] :] == three
    assert only.covers_page(4) and only.covers_page(5) and not only.covers_page(6)


def test_a_page_break_inside_a_paragraph_is_kept(cfg):
    text = sentences(2, 10, "A")
    break_at = text.index("A1x0")

    only = chunk(Doc().heading("H", page=7).para(text, page=7, marks=[(break_at, 8)]), cfg)[0]

    assert (only.page, only.page_end) == (7, 8)
    offset, page = only.page_marks[0]
    assert page == 8 and only.text[offset:].startswith("A1x0")


def test_pages_follow_the_text_into_later_chunks(cfg):
    doc = Doc()
    for page in range(1, 7):
        doc.para(sentences(4, 10, f"P{page}Q"), page=page)

    chunks = chunk(doc, cfg)

    for c in chunks:
        # The first word of the chunk says which page it came from (P3Q... is page 3).
        assert c.page == int(c.text.split()[0][1])
        for offset, page in c.page_marks:
            assert c.text[offset:].split()[0].startswith(f"P{page}Q")
        assert c.last_page == int(c.text.split()[-1][1])


# -- tables and figures ---------------------------------------------------------------


def test_a_table_is_one_chunk_however_long_it_is(cfg):
    doc = Doc().heading("Charges")
    long_table = "Table: Charges\nColumns: A | B\n" + "\n".join(f"row{n} | value{n}" for n in range(400))
    doc.table(long_table, page=3, html="<table><tr><td>row0</td></tr></table>")

    chunks = chunk(doc, cfg)
    table = next(c for c in chunks if c.element_type is ElementType.TABLE)

    assert table.text == long_table
    assert size(table) > MAX
    assert table.html == "<table><tr><td>row0</td></tr></table>"
    assert table.page == 3 and table.heading_path == ("Charges",)
    assert sum(c.element_type is ElementType.TABLE for c in chunks) == 1


def test_a_described_figure_becomes_a_model_generated_chunk(cfg):
    doc = Doc().heading("Process")
    figure = doc.figure("Figure 1: Loan application process", page=2, image="crop.png")
    description = FigureDescription(
        image_sha256="a" * 64,
        model="qwen2.5vl:7b",
        figure_type="flowchart",
        title="Loan application process",
        axes_and_legend=None,
        nodes=("Application received", "Credit score check", "Approve"),
        edges=(
            FlowchartEdge("Application received", "Credit score check"),
            FlowchartEdge("Credit score check", "Approve", "score at or above the cut-off"),
        ),
    )

    only = next(c for c in chunk(doc, cfg, {figure.element_id: description}) if c.element_type is ElementType.FIGURE)

    assert only.model_generated is True
    assert only.image_path == Path("crop.png")
    assert only.page == 2
    assert only.text.splitlines() == [
        "Figure (flowchart): Loan application process",
        "Caption: Figure 1: Loan application process",
        "Boxes: Application received; Credit score check; Approve",
        "Arrows:",
        "- Application received -> Credit score check",
        "- Credit score check -> Approve [score at or above the cut-off]",
    ]


def test_a_figure_without_a_description_is_indexed_by_its_caption_only(cfg):
    doc = Doc()
    doc.figure("Figure 2: Organisation chart", page=5)
    doc.figure("", page=6)

    figures = [c for c in chunk(doc, cfg) if c.element_type is ElementType.FIGURE]

    assert [(c.text, c.model_generated, c.page) for c in figures] == [
        ("Figure caption: Figure 2: Organisation chart", False, 5)
    ]


def test_a_logo_with_nothing_readable_is_not_indexed(cfg):
    doc = Doc()
    logo = doc.figure("", page=1)
    description = FigureDescription("b" * 64, "qwen2.5vl:7b", "logo", None, None)

    assert chunk(doc, cfg, {logo.element_id: description}) == []


def test_chunk_ids_follow_the_reading_order_of_the_document(cfg):
    doc = Doc().heading("Section A").para(sentences(5, 10, "A"))
    doc.table("Table: T\nColumns: A\nx", page=1)
    doc.heading("Section B").para(sentences(5, 10, "B"))
    doc.figure("Figure 9: Last", page=2)

    chunks = chunk(doc, cfg)

    assert [c.chunk_id for c in chunks] == [f"doc1:{n:05d}" for n in range(4)]
    assert [c.element_type.value for c in chunks] == ["text", "table", "text", "figure"]


def test_an_empty_document_gives_no_chunks(cfg):
    assert chunk(Doc(), cfg) == []


def test_without_a_counter_the_embedding_models_tokenizer_is_needed(temp_config, tmp_path):
    def model_that_is_not_there(raw):
        raw["embedding"]["model"] = str(tmp_path / "no-such-model")

    config = temp_config(model_that_is_not_there)

    with pytest.raises(TokenCounterError, match="tokenizer"):
        chunk_elements(Doc().para("Some text.").elements, INFO, {}, config)


# -- sentence boundaries --------------------------------------------------------------


def split(text: str) -> list[str]:
    return [text[start:end] for start, end in sentence_spans(text)]


def test_sentences_are_split_at_full_stops_and_line_breaks():
    assert split("The fee is 2%. It is not refunded. Is that clear? Yes.") == [
        "The fee is 2%. ",
        "It is not refunded. ",
        "Is that clear? ",
        "Yes.",
    ]
    assert split("(a) first item\n(b) second item") == ["(a) first item\n", "(b) second item"]


def test_abbreviations_numbers_and_initials_do_not_end_a_sentence():
    assert split("The limit is Rs. 5,000 per day. Circular No. 12 applies.") == [
        "The limit is Rs. 5,000 per day. ",
        "Circular No. 12 applies.",
    ]
    assert split("See para 4.2 of the circular, e.g. the table. Done.") == [
        "See para 4.2 of the circular, e.g. the table. ",
        "Done.",
    ]
    assert split("Signed by A. Kumar and others.") == ["Signed by A. Kumar and others."]


def test_sentence_spans_cover_the_whole_text():
    text = "One. Two!  Three?\n\nFour"
    spans = sentence_spans(text)

    assert "".join(text[start:end] for start, end in spans) == text
    assert sentence_spans("") == []
