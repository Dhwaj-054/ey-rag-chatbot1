"""The sample files that the real-model tests feed to Docling and the vision model.

Those tests could not be run where they were written. These fast tests at least make
sure that the files they are given are what they claim to be: a PDF with a text layer,
a PDF with none, and a picture.
"""

from __future__ import annotations

import pypdfium2 as pdfium
import pytest
from PIL import Image

from tests.pdf_samples import flowchart_png, scanned_pdf, text_pdf

LINES = ["Schedule of charges", "The processing fee on a home loan is 0.50 percent.", "Prepayment is free."]


def page_text(path) -> tuple[int, str]:
    document = pdfium.PdfDocument(str(path))
    try:
        text_page = document[0].get_textpage()
        try:
            return len(document), text_page.get_text_range()
        finally:
            text_page.close()
    finally:
        document.close()


def test_text_pdf_is_a_valid_one_page_pdf_with_a_text_layer(tmp_path):
    pages, text = page_text(text_pdf(tmp_path / "policy.pdf", LINES))

    assert pages == 1
    assert [line.strip() for line in text.splitlines() if line.strip()] == LINES


def test_text_pdf_refuses_what_its_simple_writer_cannot_encode(tmp_path):
    for line in ("Fee (see note)", "Rate is 5\\6", "Fee is ₹500"):
        with pytest.raises(ValueError, match="Not supported"):
            text_pdf(tmp_path / "bad.pdf", ["Heading", line])


def test_scanned_pdf_has_a_page_image_and_no_text_layer(tmp_path):
    pages, text = page_text(scanned_pdf(tmp_path / "scan.pdf", ["CIRCULAR ON PREPAYMENT", "No charge applies."]))

    assert pages == 1
    assert text.strip() == ""


def test_flowchart_is_a_picture_with_something_drawn_on_it(tmp_path):
    with Image.open(flowchart_png(tmp_path / "flow.png")) as image:
        assert image.size == (900, 700)
        assert image.convert("L").getextrema() == (0, 255)
