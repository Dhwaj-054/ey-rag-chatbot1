"""Stage 1, ingestion: turn the files in ``docs/`` into chunks that are ready to index.

    file -> file_sha256 -> parse_document -> describe_figure -> chunk_elements -> index

Run it from the project folder::

    python -m docqa.ingest docs/ --limit 3

The work is split over the modules of this package:

    documents   which document a file is, its hash, and its metadata (the manifest)
    parser      Docling, and the parse cache
    elements    Docling's items -> this project's elements; figure crops
    tables      the text version of a table
    screening   the check for personal identifiers
    figures     the vision model's description of a figure, and its cache
    chunker     elements -> chunks
    pipeline    all of the above in order, for one file or a folder

The five functions below are the stage's interface. Everything else is detail.
"""

from __future__ import annotations

from docqa.ingest.chunker import chunk_elements
from docqa.ingest.documents import file_sha256
from docqa.ingest.figures import describe_figure
from docqa.ingest.parser import parse_document
from docqa.ingest.pipeline import ingest_path

__all__ = [
    "chunk_elements",
    "describe_figure",
    "file_sha256",
    "ingest_path",
    "parse_document",
]
