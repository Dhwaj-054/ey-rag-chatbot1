"""Parsing a PDF with Docling, and caching the result.

Parsing is the slowest step after figure description, so the parsed document is saved
under ``paths.parsed_dir`` and read back on later runs:

    <doc_id>.docling.json   Docling's own document format, figure images included
    <doc_id>.meta.json      the file's hash and the settings it was parsed with

The cache is used only when the hash and the settings both match. Changing a chunking
setting therefore costs no parsing, while changing ``parsing.images_scale`` does.

Settings that differ from Docling's defaults, and why:

- ``generate_picture_images`` is on. Without it Docling keeps no picture crops, and
  there would be nothing to show the vision model.
- ``images_scale`` comes from the configuration. At Docling's default of 1.0 (72 dpi)
  small labels in a chart are too coarse to read.
- ``enable_remote_services`` is off and stays off: nothing leaves the machine.

Two things here are about speed:

- **OCR only where it is needed** (``parsing.ocr: auto``). Before a document is parsed,
  its pages are checked for a text layer, which takes milliseconds. A document whose
  pages all have one is parsed without OCR. A document with a page that is only a
  picture is parsed with it. A parse that was made with OCR is reused where OCR is no
  longer asked for, because it holds everything the other would.
- **Threads** (``parsing.num_threads: 0``): Docling gets one thread per physical core
  of the machine it runs on, instead of a fixed number.

NOT VERIFIED on the machine this was written on: ``convert_pdf`` itself. Docling's
layout, table and OCR models were not available there, so only the option building,
the text-layer check and the cache logic have been run. The element extraction that
follows is tested on hand-built Docling documents.
"""

from __future__ import annotations

import gc
import importlib.metadata
import json
import logging
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from docqa.config import Config
from docqa.db import utc_now
from docqa.schema import DocumentInfo, Element

if TYPE_CHECKING:
    from docling.document_converter import DocumentConverter
    from docling_core.types.doc import DoclingDocument

log = logging.getLogger(__name__)

# Raise this when the way documents are parsed changes in a way the settings below do
# not capture. Every cached parse then counts as out of date.
PARSER_VERSION = "1"

_converters: dict[str, Any] = {}
_converters_lock = threading.Lock()


class ParseError(Exception):
    """Docling could not parse the file."""


def thread_count(config: Config) -> int:
    """How many threads Docling may use: ``parsing.num_threads``, or one per physical core."""
    if config.parsing.num_threads > 0:
        return config.parsing.num_threads
    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
    except Exception:  # noqa: BLE001 - a missing or failing psutil must not stop parsing
        physical = None
    return physical or os.cpu_count() or 4


def has_scanned_page(path: Path, max_chars: int) -> bool:
    """True if at least one page of the PDF is a picture with (almost) no text layer.

    A page counts as scanned when it holds an image and at most ``max_chars``
    characters of text. An empty page holds no image and does not count. If the file
    cannot be read here, the answer is True: Docling then decides for itself.
    """
    try:
        import pypdfium2 as pdfium
        import pypdfium2.raw as pdfium_raw

        document = pdfium.PdfDocument(str(path))
    except Exception as exc:  # noqa: BLE001 - damaged or protected file
        log.info("%s could not be checked for a text layer (%s). OCR stays on.", path.name, exc)
        return True
    try:
        for number in range(len(document)):
            page = document[number]
            try:
                text_page = page.get_textpage()
                try:
                    characters = text_page.count_chars()
                finally:
                    text_page.close()
                if characters > max_chars:
                    continue
                images = page.get_objects(filter=(pdfium_raw.FPDF_PAGEOBJ_IMAGE,), max_depth=3)
                if next(iter(images), None) is not None:
                    return True
            finally:
                page.close()
        return False
    except Exception as exc:  # noqa: BLE001
        log.info("%s could not be checked for a text layer (%s). OCR stays on.", path.name, exc)
        return True
    finally:
        document.close()


def needs_ocr(path: Path, config: Config) -> bool:
    """Whether this document is parsed with OCR, according to ``parsing.ocr``."""
    if config.parsing.ocr == "auto":
        return has_scanned_page(path, config.parsing.scanned_page_max_chars)
    return bool(config.parsing.ocr)


def parse_settings(config: Config, ocr: bool | None = None) -> dict[str, Any]:
    """Everything that decides what a parse looks like.

    Without ``ocr`` it describes the configuration and is part of each document's
    fingerprint. With ``ocr`` it describes one actual parse and is stored next to the
    cached result.
    """
    parsing = config.parsing
    return {
        "parser_version": PARSER_VERSION,
        "docling": _package_version("docling"),
        "ocr": parsing.ocr if ocr is None else ocr,
        "ocr_languages": list(parsing.ocr_languages),
        "images_scale": parsing.images_scale,
        "table_mode": parsing.table_mode,
    }


def build_pipeline_options(config: Config, ocr: bool = True) -> Any:
    """Docling's PDF options, set from the ``parsing`` section of the configuration."""
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.pipeline_options import (
        OcrAutoOptions,
        PdfPipelineOptions,
        TableFormerMode,
        TableStructureOptions,
    )

    parsing = config.parsing
    return PdfPipelineOptions(
        artifacts_path=str(parsing.artifacts_path) if parsing.artifacts_path else None,
        enable_remote_services=False,
        do_ocr=ocr,
        ocr_options=OcrAutoOptions(lang=list(parsing.ocr_languages)),
        do_table_structure=True,
        table_structure_options=TableStructureOptions(
            mode=TableFormerMode(parsing.table_mode), do_cell_matching=True
        ),
        generate_picture_images=True,
        images_scale=parsing.images_scale,
        accelerator_options=AcceleratorOptions(num_threads=thread_count(config)),
    )


def get_converter(config: Config, ocr: bool = True) -> DocumentConverter:
    """A Docling converter for PDFs. Built once per set of settings and reused.

    Building it is cheap. Docling loads its models the first time a document is
    converted, which takes a while, so one converter serves a whole ingestion run.
    """
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption

    key = json.dumps(
        {
            **parse_settings(config, ocr),
            "artifacts_path": str(config.parsing.artifacts_path),
            "threads": thread_count(config),
        },
        sort_keys=True,
    )
    with _converters_lock:
        if key not in _converters:
            _converters[key] = DocumentConverter(
                allowed_formats=[InputFormat.PDF],
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=build_pipeline_options(config, ocr))
                },
            )
        return _converters[key]


def release_converters() -> None:
    """Forget the converters, so that Docling's models stop taking up memory.

    Ingestion calls this once a group of files has been parsed, before the vision
    model and the embedding model take their turn.
    """
    with _converters_lock:
        _converters.clear()
    gc.collect()


def convert_pdf(path: Path, config: Config, ocr: bool = True) -> DoclingDocument:
    """Run Docling on one PDF.

    Raises:
        ParseError: Docling reported a failure, or could not be run at all (for
            example because its models are not on this machine).
    """
    from docling.datamodel.base_models import ConversionStatus

    try:
        result = get_converter(config, ocr).convert(path, raises_on_error=False)
    except Exception as exc:  # missing models, a corrupt file, out of memory
        raise ParseError(f"Docling could not parse {path.name}: {type(exc).__name__}: {exc}") from exc

    problems = "; ".join(str(getattr(error, "error_message", error)) for error in result.errors)
    if result.status == ConversionStatus.PARTIAL_SUCCESS:
        log.warning("%s was parsed only in part: %s", path.name, problems or "no detail given")
    elif result.status != ConversionStatus.SUCCESS:
        raise ParseError(
            f"Docling could not parse {path.name} (status {result.status.value}): "
            f"{problems or 'no detail given'}"
        )
    return result.document


def cache_paths(doc_id: str, config: Config) -> tuple[Path, Path]:
    """Where the cached parse of a document lives: ``(document file, metadata file)``."""
    folder = config.paths.parsed_dir
    return folder / f"{doc_id}.docling.json", folder / f"{doc_id}.meta.json"


def _cache_is_usable(meta: dict[str, Any], sha256: str, wanted: dict[str, Any]) -> bool:
    """Whether a cached parse can stand in for the one that is wanted now."""
    if meta.get("sha256") != sha256 or not isinstance(meta.get("settings"), dict):
        return False
    cached = dict(meta["settings"])
    if cached == wanted:
        return True
    # A parse made with OCR holds everything a parse without it would.
    if cached.get("ocr") is True and wanted.get("ocr") is False:
        cached["ocr"] = False
        return cached == wanted
    return False


def load_parsed(info: DocumentInfo, config: Config) -> DoclingDocument:
    """The parsed document: from the cache if it is current, otherwise parsed now and cached."""
    from docling_core.types.doc import DoclingDocument

    document_file, meta_file = cache_paths(info.doc_id, config)
    ocr = needs_ocr(info.source_path, config)
    settings = parse_settings(config, ocr)

    if document_file.is_file() and meta_file.is_file():
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            if _cache_is_usable(meta, info.sha256, settings):
                return DoclingDocument.load_from_json(document_file)
        except Exception as exc:  # a damaged cache file is not a reason to stop
            log.warning("Cached parse of %s could not be read (%s). Parsing again.", info.doc_title, exc)

    document = convert_pdf(info.source_path, config, ocr)
    document_file.parent.mkdir(parents=True, exist_ok=True)
    document.save_as_json(document_file)
    meta_file.write_text(
        json.dumps(
            {"sha256": info.sha256, "settings": settings, "parsed_at": utc_now(), "source": info.source_path.name},
            indent=2,
        ),
        encoding="utf-8",
    )
    return document


def parse_document(info: DocumentInfo, config: Config) -> list[Element]:
    """Parse one document with Docling and return its elements in reading order.

    Every element keeps its ``doc_id``, page number, bounding box and element type.
    A table keeps its original as HTML, plus a text version that includes its title
    and column headers. A figure is cropped and saved under ``paths.figures_dir``.
    Scanned pages go through Docling's built-in OCR.

    The parse result is cached under ``paths.parsed_dir``, so an unchanged file is
    never parsed twice.
    """
    from docqa.ingest.elements import extract_elements

    return extract_elements(load_parsed(info, config), info, config).elements


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"
