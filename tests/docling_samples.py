"""Docling documents built by hand, standing in for what Docling returns for a PDF.

Docling's layout, table and OCR models are not needed to build a ``DoclingDocument``:
the document model is plain data. That lets the code that reads Docling's output be
tested against the real data model, without parsing a real PDF.
"""

from __future__ import annotations

from docling_core.types.doc import (
    BoundingBox,
    ContentLayer,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    ImageRef,
    ProvenanceItem,
    Size,
    TableCell,
    TableData,
)
from PIL import Image

PAGE_WIDTH, PAGE_HEIGHT = 595.0, 842.0


def prov(page: int, top: float, bottom: float, left: float = 50, right: float = 545, length: int = 0):
    """Where an item sits. Coordinates are in points from the bottom left, as in a PDF."""
    box = BoundingBox(l=left, t=top, r=right, b=bottom, coord_origin=CoordOrigin.BOTTOMLEFT)
    return ProvenanceItem(page_no=page, bbox=box, charspan=(0, length))


def table_data(rows: list[list[str]], header_rows: int = 1) -> TableData:
    cells = [
        TableCell(
            text=text,
            start_row_offset_idx=r,
            end_row_offset_idx=r + 1,
            start_col_offset_idx=c,
            end_col_offset_idx=c + 1,
            column_header=r < header_rows,
        )
        for r, row in enumerate(rows)
        for c, text in enumerate(row)
    ]
    return TableData(table_cells=cells, num_rows=len(rows), num_cols=len(rows[0]))


FEES_PARAGRAPH = (
    "The bank charges a processing fee on every retail loan. The fee is collected "
    "when the loan is sanctioned and is not refunded."
)
SPLIT_PARAGRAPH_FIRST = "A borrower may repay the loan before it is due. "
SPLIT_PARAGRAPH_SECOND = "No charge applies to a floating rate home loan."
CHARGES = [
    ["Loan type", "Processing fee", "Prepayment charge"],
    ["Home loan", "0.50% of the loan amount", "Nil"],
    ["Personal loan", "2.00% of the loan amount", "3% of the outstanding principal"],
]


def lending_policy() -> DoclingDocument:
    """A two-page policy: title, two sections, a list, a table, a flowchart and a logo."""
    doc = DoclingDocument(name="lending-policy")
    for page in (1, 2):
        doc.add_page(page_no=page, size=Size(width=PAGE_WIDTH, height=PAGE_HEIGHT))

    doc.add_text(
        DocItemLabel.PAGE_HEADER,
        "Example Bank - Lending Policy",
        prov=prov(1, 830, 815),
        content_layer=ContentLayer.FURNITURE,
    )
    # A small picture with no caption, such as a logo in the corner of the page.
    doc.add_picture(
        image=ImageRef.from_pil(Image.new("RGB", (80, 40), "navy"), dpi=144),
        prov=prov(1, 830, 800, left=480, right=545),
    )
    doc.add_title("Lending Policy", prov=prov(1, 790, 770))
    doc.add_heading("1. Fees and charges", level=1, prov=prov(1, 750, 735))
    doc.add_text(DocItemLabel.TEXT, FEES_PARAGRAPH, prov=prov(1, 725, 690))

    caption = doc.add_text(
        DocItemLabel.CAPTION, "Table 1: Schedule of charges for retail loans", prov=prov(1, 680, 668)
    )
    doc.add_table(data=table_data(CHARGES), caption=caption, prov=prov(1, 660, 580))

    doc.add_heading("2. Prepayment", level=1, prov=prov(1, 560, 545))
    conditions = doc.add_list_group(name="conditions")
    doc.add_list_item(
        "the borrower gives thirty days' notice;",
        enumerated=True,
        marker="(a)",
        parent=conditions,
        prov=prov(1, 535, 520),
    )
    doc.add_list_item(
        "all dues up to the date of repayment are cleared.",
        enumerated=True,
        marker="(b)",
        parent=conditions,
        prov=prov(1, 515, 500),
    )
    # One paragraph that starts at the foot of page 1 and ends at the top of page 2.
    running = doc.add_text(
        DocItemLabel.TEXT,
        SPLIT_PARAGRAPH_FIRST + SPLIT_PARAGRAPH_SECOND,
        prov=prov(1, 90, 60, length=len(SPLIT_PARAGRAPH_FIRST)),
    )
    second_part = prov(2, 800, 780)
    second_part.charspan = (len(SPLIT_PARAGRAPH_FIRST), len(SPLIT_PARAGRAPH_FIRST + SPLIT_PARAGRAPH_SECOND))
    running.prov.append(second_part)
    doc.add_text(
        DocItemLabel.PAGE_FOOTER, "Page 1 of 2", prov=prov(1, 40, 28), content_layer=ContentLayer.FURNITURE
    )

    figure_caption = doc.add_text(
        DocItemLabel.CAPTION, "Figure 1: Loan application process", prov=prov(2, 470, 458)
    )
    doc.add_picture(
        image=ImageRef.from_pil(Image.new("RGB", (600, 400), "white"), dpi=144),
        caption=figure_caption,
        prov=prov(2, 760, 480),
    )
    doc.add_text(DocItemLabel.FOOTNOTE, "Charges are exclusive of taxes.", prov=prov(2, 120, 105))
    return doc


def with_table_of_contents() -> DoclingDocument:
    """A document whose first table is its table of contents."""
    doc = DoclingDocument(name="with-contents")
    doc.add_page(page_no=1, size=Size(width=PAGE_WIDTH, height=PAGE_HEIGHT))
    doc.add_table(
        data=table_data([["Section", "Page"], ["1. Fees", "2"]]),
        prov=prov(1, 700, 600),
        label=DocItemLabel.DOCUMENT_INDEX,
    )
    doc.add_text(DocItemLabel.TEXT, "The only real sentence in this document.", prov=prov(1, 500, 480))
    return doc


def with_identifiers(count: int) -> DoclingDocument:
    """A document that quotes ``count`` different PAN-like strings."""
    doc = DoclingDocument(name="identifiers")
    doc.add_page(page_no=1, size=Size(width=PAGE_WIDTH, height=PAGE_HEIGHT))
    for number in range(count):
        doc.add_text(
            DocItemLabel.TEXT,
            f"Customer record {number}: PAN ABCPE{1000 + number}F, outstanding balance 1,20,000.",
            prov=prov(1, 700 - 20 * number, 685 - 20 * number),
        )
    return doc
