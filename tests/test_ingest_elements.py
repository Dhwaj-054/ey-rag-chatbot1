"""From Docling's document to this project's elements, on hand-built Docling documents."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from docling_core.types.doc import DoclingDocument
from PIL import Image

from docqa.ingest.elements import extract_elements
from docqa.ingest.tables import UNTITLED, clean_cell, column_headers, table_to_text
from docqa.schema import DocumentInfo, ElementType, TextRole
from tests.docling_samples import (
    FEES_PARAGRAPH,
    SPLIT_PARAGRAPH_FIRST,
    SPLIT_PARAGRAPH_SECOND,
    lending_policy,
    with_decorations,
    with_table_of_contents,
)


def make_info(doc_id: str = "policy01") -> DocumentInfo:
    return DocumentInfo(
        doc_id=doc_id,
        doc_title="Lending Policy",
        source_path=Path("Lending Policy.pdf"),
        sha256="0" * 64,
        version=None,
        effective_date=date(2026, 4, 1),
        access_group="all",
    )


@pytest.fixture
def extraction(tmp_cfg):
    return extract_elements(lending_policy(), make_info(), tmp_cfg)


def test_elements_come_out_in_reading_order_with_their_roles(extraction):
    summary = [
        (element.element_type.value, element.role.value if element.element_type is ElementType.TEXT else "-")
        for element in extraction.elements
    ]

    assert summary == [
        ("text", "title"),
        ("text", "heading"),
        ("text", "paragraph"),
        ("table", "-"),
        ("text", "heading"),
        ("text", "list_item"),
        ("text", "list_item"),
        ("text", "paragraph"),
        ("figure", "-"),
        ("text", "footnote"),
    ]
    assert extraction.pages == 2
    assert [element.element_id for element in extraction.elements[:2]] == ["policy01:e00000", "policy01:e00001"]
    assert all(element.doc_id == "policy01" for element in extraction.elements)


def test_page_headers_footers_and_captions_are_not_repeated_as_text(extraction):
    texts = [element.text for element in extraction.elements if element.element_type is ElementType.TEXT]

    assert not any("Example Bank - Lending Policy" in text for text in texts)
    assert not any("Page 1 of 2" in text for text in texts)
    # Each caption belongs to its table or figure and appears only there.
    assert not any(text.startswith(("Table 1:", "Figure 1:")) for text in texts)


def test_each_element_knows_the_headings_above_it(extraction):
    by_text = {element.text: element for element in extraction.elements}

    assert by_text["Lending Policy"].heading_path == ()
    assert by_text["1. Fees and charges"].heading_path == ()
    assert by_text["1. Fees and charges"].heading_level == 1
    assert by_text[FEES_PARAGRAPH].heading_path == ("1. Fees and charges",)
    assert by_text["Charges are exclusive of taxes."].heading_path == ("2. Prepayment",)


def test_list_items_keep_their_markers_and_share_a_list_id(extraction):
    items = [element for element in extraction.elements if element.role is TextRole.LIST_ITEM]

    assert [item.text for item in items] == [
        "(a) the borrower gives thirty days' notice;",
        "(b) all dues up to the date of repayment are cleared.",
    ]
    assert items[0].list_id is not None
    assert items[0].list_id == items[1].list_id
    assert items[0].heading_path == ("2. Prepayment",)


def test_a_table_is_kept_as_html_and_as_text_with_title_and_headers(extraction):
    table = next(element for element in extraction.elements if element.element_type is ElementType.TABLE)

    assert table.text.splitlines() == [
        "Table: Table 1: Schedule of charges for retail loans",
        "Columns: Loan type | Processing fee | Prepayment charge",
        "Home loan | 0.50% of the loan amount | Nil",
        "Personal loan | 2.00% of the loan amount | 3% of the outstanding principal",
    ]
    assert table.html.startswith("<table>")
    assert "<th>Processing fee</th>" in table.html
    assert "<td>0.50% of the loan amount</td>" in table.html
    assert table.page == 1
    assert table.heading_path == ("1. Fees and charges",)


def test_a_figure_is_cropped_to_a_file_and_carries_its_caption(extraction, tmp_cfg):
    figure = next(element for element in extraction.elements if element.element_type is ElementType.FIGURE)

    assert figure.text == "Figure 1: Loan application process"
    assert figure.page == 2
    assert figure.image_path == tmp_cfg.paths.figures_dir / "policy01" / "p0002-f001.png"
    with Image.open(figure.image_path) as image:
        assert image.size == (600, 400)


def test_a_small_picture_without_a_caption_is_left_out(extraction, tmp_cfg):
    """The logo covers well under figures.min_page_fraction of its page."""
    assert extraction.small_figures_skipped == 1
    assert len(list((tmp_cfg.paths.figures_dir / "policy01").glob("*.png"))) == 1


def test_banners_and_stamps_are_left_out_unless_they_have_a_caption(tmp_cfg):
    """Each picture left out is a call to the vision model saved."""
    extraction = extract_elements(with_decorations(), make_info("deco0001"), tmp_cfg)

    figures = [element for element in extraction.elements if element.element_type is ElementType.FIGURE]
    sizes = []
    for figure in figures:
        with Image.open(figure.image_path) as crop:
            sizes.append(crop.size)

    assert extraction.small_figures_skipped == 2
    assert sizes == [(600, 400), (1000, 60)]
    assert [figure.text for figure in figures] == ["", "Figure 2: Timeline of the scheme"]
    assert len(list((tmp_cfg.paths.figures_dir / "deco0001").glob("*.png"))) == 2


def test_a_paragraph_that_runs_over_a_page_break_records_where(extraction):
    running = next(
        element for element in extraction.elements if element.text.startswith("A borrower may repay")
    )

    assert running.page == 1
    assert running.page_marks == ((len(SPLIT_PARAGRAPH_FIRST), 2),)
    assert running.text[running.page_marks[0][0] :] == SPLIT_PARAGRAPH_SECOND


def test_bounding_boxes_are_measured_from_the_top_of_the_page(extraction):
    title = extraction.elements[0]

    # The title was placed 790 to 770 points above the bottom of an 842-point page.
    assert (title.bbox.left, title.bbox.right) == (50, 545)
    assert title.bbox.top == pytest.approx(842 - 790)
    assert title.bbox.bottom == pytest.approx(842 - 770)


def test_extraction_gives_the_same_result_after_the_parse_cache_round_trip(tmp_path, tmp_cfg):
    """The parse cache stores Docling's JSON. Reading it back must change nothing."""
    cached = tmp_path / "cached.docling.json"
    lending_policy().save_as_json(cached)
    reloaded = DoclingDocument.load_from_json(cached)

    direct = extract_elements(lending_policy(), make_info("a"), tmp_cfg).elements
    from_cache = extract_elements(reloaded, make_info("a"), tmp_cfg).elements

    assert from_cache == direct


def test_crops_of_an_earlier_version_of_the_document_are_removed(tmp_cfg):
    folder = tmp_cfg.paths.figures_dir / "policy01"
    folder.mkdir(parents=True)
    (folder / "p0009-f009.png").write_bytes(b"left over from the previous version")

    extract_elements(lending_policy(), make_info(), tmp_cfg)

    assert sorted(path.name for path in folder.iterdir()) == ["p0002-f001.png"]


def test_a_table_of_contents_is_not_turned_into_a_chunk(tmp_cfg):
    elements = extract_elements(with_table_of_contents(), make_info(), tmp_cfg).elements

    assert [element.text for element in elements] == ["The only real sentence in this document."]


# -- the text version of a table ------------------------------------------------------


def test_table_text_starts_with_title_and_column_headers():
    text = table_to_text("Schedule of charges", [["Loan", "Fee"], ["Home loan", "0.50%"]], header_rows=1)

    assert text == "Table: Schedule of charges\nColumns: Loan | Fee\nHome loan | 0.50%"


def test_a_table_without_marked_headers_uses_its_first_row():
    text = table_to_text(None, [["Loan", "Fee"], ["Home loan", "0.50%"]], header_rows=0)

    assert text == f"Table: {UNTITLED}\nColumns: Loan | Fee\nHome loan | 0.50%"


def test_headers_over_several_rows_are_joined_top_down():
    rows = [
        ["Loan type", "Charges", "Charges"],
        ["Loan type", "Processing fee", "Prepayment"],
        ["Home loan", "0.50%", "Nil"],
    ]

    assert column_headers(rows, 2) == ["Loan type", "Charges / Processing fee", "Charges / Prepayment"]
    assert table_to_text("T", rows, 2).splitlines()[1] == (
        "Columns: Loan type | Charges / Processing fee | Charges / Prepayment"
    )


def test_cells_are_single_lines_and_cannot_break_a_column():
    assert clean_cell("  0.50%\nof the   loan | amount ") == "0.50% of the loan / amount"
    assert table_to_text("T", [["A"], ["", ], ["x"]], 1) == "Table: T\nColumns: A\nx"
