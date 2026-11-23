"""From a parsed Docling document to this project's elements.

Docling returns a tree of items: titles, section headings, paragraphs, list items,
tables, pictures, captions, footnotes and page furniture. This module walks it in
reading order and keeps one ``Element`` per item that carries content:

- running text, with its role (heading, paragraph, list item, footnote), the headings
  above it, and the list it belongs to, which is what chunking needs;
- tables, as HTML plus a text version that starts with the title and column headers;
- pictures, cropped to an image file under ``paths.figures_dir``.

Left out on purpose: page headers and footers, a caption that already belongs to a
table or picture (it becomes that table's or picture's title), the table of contents,
and pictures that are small and have no caption (logos, stamps, signatures).

Nothing here runs a model, so it is tested on Docling documents built by hand.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from docqa.config import Config
from docqa.ingest.tables import table_to_text
from docqa.schema import BoundingBox, DocumentInfo, Element, ElementType, PageMark, TextRole

if TYPE_CHECKING:
    from docling_core.types.doc import DoclingDocument

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Extraction:
    """The elements of one document, and what was left out."""

    elements: list[Element]
    pages: int
    small_figures_skipped: int = 0
    """Pictures below ``figures.min_page_fraction`` with no caption."""
    figures_without_image: int = 0
    """Pictures Docling found but returned no image for."""


def extract_elements(doc: DoclingDocument, info: DocumentInfo, config: Config) -> Extraction:
    """Turn one parsed document into elements, in reading order.

    Figure crops are written to ``paths.figures_dir/<doc_id>/``. Crops left there by an
    earlier version of the same document are removed first.
    """
    from docling_core.types.doc import (
        DocItemLabel,
        ListItem,
        PictureItem,
        SectionHeaderItem,
        TableItem,
        TextItem,
        TitleItem,
    )

    figures_dir = config.paths.figures_dir / info.doc_id
    if figures_dir.exists():
        shutil.rmtree(figures_dir, ignore_errors=True)

    items = [item for item, _level in doc.iterate_items()]
    caption_refs = {
        ref.cref for item in items for ref in (getattr(item, "captions", None) or [])
    }
    furniture = {DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER}

    elements: list[Element] = []
    headings: list[tuple[int, str]] = []  # (level, text) of the headings currently open
    last_page = 1
    figures = small_skipped = without_image = 0

    def add(element_type: ElementType, text: str, page: int, bbox: BoundingBox | None, **more: Any) -> None:
        elements.append(
            Element(
                doc_id=info.doc_id,
                element_id=f"{info.doc_id}:e{len(elements):05d}",
                page=page,
                bbox=bbox,
                element_type=element_type,
                text=text,
                **more,
            )
        )

    for item in items:
        label = getattr(item, "label", None)
        if label in furniture:
            continue
        provenance = list(getattr(item, "prov", None) or [])
        page = provenance[0].page_no if provenance else last_page
        last_page = page
        bbox = _bounding_box(doc, provenance[0]) if provenance else None
        path = tuple(text for _level, text in headings)

        if isinstance(item, TableItem):
            if label == DocItemLabel.DOCUMENT_INDEX:
                continue  # a table of contents matches every question and answers none
            rows = [[cell.text for cell in row] for row in item.data.grid]
            if not any(cell.strip() for row in rows for cell in row):
                continue
            header_rows = 0
            for row in item.data.grid:
                if row and all(cell.column_header for cell in row):
                    header_rows += 1
                else:
                    break
            caption = _one_line(item.caption_text(doc))
            title = caption or (path[-1] if path else None)
            add(
                ElementType.TABLE,
                table_to_text(title, rows, header_rows),
                page,
                bbox,
                heading_path=path,
                html=item.export_to_html(doc=doc, add_caption=True),
                page_marks=_later_pages(provenance),
            )

        elif isinstance(item, PictureItem):
            caption = _one_line(item.caption_text(doc))
            fraction = _page_fraction(doc, provenance[0]) if provenance else None
            if (
                not caption
                and fraction is not None
                and fraction < config.figures.min_page_fraction
            ):
                small_skipped += 1
                continue
            image = item.get_image(doc)
            image_path: Path | None = None
            if image is None:
                without_image += 1
                if not caption:
                    continue
            else:
                figures += 1
                figures_dir.mkdir(parents=True, exist_ok=True)
                image_path = figures_dir / f"p{page:04d}-f{figures:03d}.png"
                image.save(image_path, format="PNG")
            add(ElementType.FIGURE, caption, page, bbox, heading_path=path, image_path=image_path)

        elif isinstance(item, TextItem):
            if item.self_ref in caption_refs:
                continue
            text = item.text.strip()
            if not text:
                continue
            marks = _text_marks(provenance, len(text))
            if isinstance(item, TitleItem):
                add(ElementType.TEXT, _one_line(text), page, bbox, role=TextRole.TITLE)
            elif isinstance(item, SectionHeaderItem):
                level = int(getattr(item, "level", 1) or 1)
                while headings and headings[-1][0] >= level:
                    headings.pop()
                heading = _one_line(text)
                add(
                    ElementType.TEXT,
                    heading,
                    page,
                    bbox,
                    heading_path=tuple(text for _level, text in headings),
                    role=TextRole.HEADING,
                    heading_level=level,
                )
                headings.append((level, heading))
            elif isinstance(item, ListItem):
                prefix = _list_prefix(item, text)
                add(
                    ElementType.TEXT,
                    prefix + text,
                    page,
                    bbox,
                    heading_path=path,
                    role=TextRole.LIST_ITEM,
                    list_id=item.parent.cref if item.parent else None,
                    page_marks=tuple((offset + len(prefix), mark_page) for offset, mark_page in marks),
                )
            else:
                if label == DocItemLabel.CHECKBOX_SELECTED:
                    text, marks = "[x] " + text, tuple((o + 4, p) for o, p in marks)
                elif label == DocItemLabel.CHECKBOX_UNSELECTED:
                    text, marks = "[ ] " + text, tuple((o + 4, p) for o, p in marks)
                role = TextRole.FOOTNOTE if label == DocItemLabel.FOOTNOTE else TextRole.PARAGRAPH
                add(ElementType.TEXT, text, page, bbox, heading_path=path, role=role, page_marks=marks)

    if small_skipped:
        log.info("%s: skipped %d small picture(s) without a caption", info.doc_title, small_skipped)
    if without_image:
        log.warning("%s: Docling returned no image for %d picture(s)", info.doc_title, without_image)
    return Extraction(
        elements=elements,
        pages=len(doc.pages),
        small_figures_skipped=small_skipped,
        figures_without_image=without_image,
    )


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _list_prefix(item: Any, text: str) -> str:
    """What to put in front of a list item so that it still reads as one.

    A numbered item keeps its own marker, "(a)" or "3.", because regulations are cited
    by those. A bullet becomes "- ". Nothing is added when the text already starts
    with the marker.
    """
    marker = (getattr(item, "marker", None) or "").strip()
    if getattr(item, "enumerated", False) and marker:
        return "" if text.startswith(marker) else f"{marker} "
    return "" if text.startswith(("-", "•", "*")) else "- "


def _later_pages(provenance: list[Any]) -> tuple[PageMark, ...]:
    """For a table or picture that Docling places on more than one page.

    There is no position inside the text to attach to, so each later page is marked at
    offset 0. Chunking only uses these to work out the last page.
    """
    pages = sorted({entry.page_no for entry in provenance})
    return tuple((0, page) for page in pages[1:])


def _text_marks(provenance: list[Any], length: int) -> tuple[PageMark, ...]:
    """Where a text item crosses onto a later page.

    Docling gives one provenance entry per page for a paragraph that runs over a page
    break, each with the character span it covers.
    """
    marks: list[PageMark] = []
    current = provenance[0].page_no if provenance else None
    for entry in provenance[1:]:
        offset = min(max(int(entry.charspan[0]), 0), length)
        if entry.page_no != current and offset > 0:
            marks.append((offset, entry.page_no))
        current = entry.page_no
    return tuple(marks)


def _page_size(doc: Any, page_no: int) -> tuple[float, float] | None:
    page = doc.pages.get(page_no)
    if page is None or not page.size.width or not page.size.height:
        return None
    return float(page.size.width), float(page.size.height)


def _bounding_box(doc: Any, entry: Any) -> BoundingBox:
    """The item's box with the origin at the top left of the page, where the page size is known."""
    box = entry.bbox
    size = _page_size(doc, entry.page_no)
    if size is not None:
        box = box.to_top_left_origin(size[1])
    return BoundingBox(left=box.l, top=box.t, right=box.r, bottom=box.b)


def _page_fraction(doc: Any, entry: Any) -> float | None:
    """How much of its page an item covers, from 0 to 1. ``None`` if the page size is unknown."""
    size = _page_size(doc, entry.page_no)
    if size is None:
        return None
    box = entry.bbox
    return abs(box.r - box.l) * abs(box.t - box.b) / (size[0] * size[1])
