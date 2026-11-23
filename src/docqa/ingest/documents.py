"""Which document a file is, and the metadata its chunks will carry.

A document is identified by where its file sits under ``paths.docs_dir``, not by its
content. That is what lets an edited file replace its earlier chunks: same place, same
``doc_id``, new hash. Renaming or moving a file therefore makes it a new document.

The documents may also be kept in a folder somewhere else (a copy of the documents
folder on another drive, the folder that uploads go to). A file there is named by its
place below the folder it is named relative to, its *source root*:

- anything inside ``paths.docs_dir``: ``paths.docs_dir`` itself, whichever part of it
  a run is given, so that a document keeps its identity;
- a folder anywhere else: that folder. For one file given on its own: its folder;
- the uploads folder: the folder above it, so that an uploaded file is named
  ``uploads/<file>`` and can never be mistaken for a file of the documents folder.

``source_root_for`` decides the first two. The uploads folder's root is passed in by
the code that ingests uploads.

The title, version, effective date and access group come from the manifest
(``paths.manifest``), a CSV file with one row per document::

    file,doc_title,version,effective_date,access_group
    Master Circulars/Example.pdf,Master Circular - Example,2026-27/12,2026-04-01,all

``file`` is the path below ``docs_dir``. Only ``file`` is required; an empty cell means
"use the default". A document that is not listed gets its file name as its title, no
version, no effective date and ``retrieval.default_access_group``. The manifest is
optional: without it every document gets those defaults.
"""

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from docqa.config import Config
from docqa.schema import DocumentInfo

MANIFEST_COLUMNS = ("file", "doc_title", "version", "effective_date", "access_group")
_DATE_FORMATS = ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y")
_HASH_BLOCK = 1024 * 1024


class ManifestError(Exception):
    """The manifest file cannot be read as described above."""


def file_sha256(path: Path) -> str:
    """Return the SHA-256 hash of a file's contents, as lowercase hex.

    The hash decides whether a file needs any work: a file whose hash matches the one
    stored from the previous run is skipped.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(_HASH_BLOCK):
            digest.update(block)
    return digest.hexdigest()


def source_path_for(path: Path, docs_dir: Path) -> str:
    """How a file is named in the register and the manifest.

    For a file inside ``docs_dir`` this is its path below that folder, with forward
    slashes. For a file anywhere else it is the file's full path, so that two files of
    the same name in different places stay two documents.
    """
    resolved = path.resolve()
    try:
        return resolved.relative_to(docs_dir.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def source_root_for(path: Path, docs_dir: Path) -> Path:
    """The folder that the files of a run over ``path`` are named relative to.

    Inside ``docs_dir`` that is always ``docs_dir``: ingesting one of its sub-folders,
    or one of its files, must give each document the name it has when the whole folder
    is ingested. Anywhere else it is the folder given (for a file: the folder it is
    in), which is how a copy of the documents folder kept elsewhere matches the
    documents that were ingested from the original.
    """
    resolved = path.resolve()
    docs = docs_dir.resolve()
    if resolved == docs or docs in resolved.parents:
        return docs
    return resolved if resolved.is_dir() else resolved.parent


def doc_id_for(source_path: str) -> str:
    """A short, stable identifier made from the document's place under its source root."""
    return hashlib.sha256(_key(source_path).encode("utf-8")).hexdigest()[:16]


def clean_title(text: str) -> str:
    """Make a title safe to use inside a citation.

    Citations are written ``[document title, p. N]``. A title that itself contains
    square brackets would make that impossible to read back, so they become round
    ones. Line breaks and repeated spaces become single spaces.
    """
    text = text.replace("[", "(").replace("]", ")")
    return re.sub(r"\s+", " ", text).strip()


def default_title(path: Path) -> str:
    """The title of a document that the manifest does not list: its file name, tidied."""
    return clean_title(path.stem.replace("_", " ")) or path.name


@dataclass(frozen=True)
class ManifestEntry:
    """One manifest row. ``None`` means the cell was empty."""

    doc_title: str | None = None
    version: str | None = None
    effective_date: date | None = None
    access_group: str | None = None


class Manifest:
    """The rows of the manifest file, looked up by file."""

    def __init__(self, entries: dict[str, ManifestEntry] | None = None) -> None:
        self._entries = {_key(name): entry for name, entry in (entries or {}).items()}
        by_name: dict[str, list[str]] = {}
        for key in self._entries:
            by_name.setdefault(key.rsplit("/", 1)[-1], []).append(key)
        # A bare file name finds its row only if exactly one row has that file name.
        self._unique_names = {name: keys[0] for name, keys in by_name.items() if len(keys) == 1}

    def __len__(self) -> int:
        return len(self._entries)

    def lookup(self, source_path: str) -> ManifestEntry | None:
        key = _key(source_path)
        if key in self._entries:
            return self._entries[key]
        fallback = self._unique_names.get(key.rsplit("/", 1)[-1])
        return self._entries[fallback] if fallback else None

    @classmethod
    def load(cls, path: Path) -> Manifest:
        """Read the manifest. A file that does not exist gives an empty manifest.

        Raises:
            ManifestError: A column is missing or unknown, a file is listed twice, or
                a date cannot be read. The message names the row.
        """
        if not path.is_file():
            return cls()
        entries: dict[str, ManifestEntry] = {}
        try:
            with open(path, encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                columns = [name.strip() for name in reader.fieldnames or []]
                if "file" not in columns:
                    raise ManifestError(f"{path}: the header row must have a 'file' column.")
                unknown = [name for name in columns if name not in MANIFEST_COLUMNS]
                if unknown:
                    raise ManifestError(
                        f"{path}: unknown column(s) {', '.join(unknown)}. "
                        f"Allowed: {', '.join(MANIFEST_COLUMNS)}."
                    )
                for number, row in enumerate(reader, start=2):
                    cells = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
                    name = cells.get("file", "")
                    if not name:
                        if any(cells.values()):
                            raise ManifestError(f"{path}, row {number}: 'file' is empty.")
                        continue
                    if _key(name) in {_key(seen) for seen in entries}:
                        raise ManifestError(f"{path}, row {number}: {name!r} is listed twice.")
                    entries[name] = ManifestEntry(
                        doc_title=clean_title(cells["doc_title"]) if cells.get("doc_title") else None,
                        version=cells.get("version") or None,
                        effective_date=_parse_date(cells.get("effective_date", ""), path, number),
                        access_group=cells.get("access_group") or None,
                    )
        except (OSError, csv.Error, UnicodeDecodeError) as exc:
            raise ManifestError(f"{path} could not be read: {exc}") from exc
        return cls(entries)


def document_info(
    path: Path,
    config: Config,
    manifest: Manifest,
    sha256: str | None = None,
    source_root: Path | None = None,
) -> DocumentInfo:
    """Build the record for one file: its identity, its hash and its metadata.

    ``source_root`` is the folder the file is named relative to (see the top of this
    module). Left out, it is ``paths.docs_dir``.
    """
    source_path = source_path_for(path, source_root or config.paths.docs_dir)
    entry = manifest.lookup(source_path) or ManifestEntry()
    return DocumentInfo(
        doc_id=doc_id_for(source_path),
        doc_title=entry.doc_title or default_title(path),
        source_path=path,
        sha256=sha256 or file_sha256(path),
        version=entry.version,
        effective_date=entry.effective_date,
        access_group=entry.access_group or config.retrieval.default_access_group,
    )


def source_key(source_path: str) -> str:
    """A source path in the form two of them are compared in: forward slashes, lower case."""
    return _key(source_path)


def _key(source_path: str) -> str:
    return source_path.replace("\\", "/").strip().strip("/").lower()


def _parse_date(text: str, path: Path, row: int) -> date | None:
    if not text:
        return None
    for date_format in _DATE_FORMATS:
        try:
            return datetime.strptime(text, date_format).date()
        except ValueError:
            continue
    raise ManifestError(
        f"{path}, row {row}: effective_date {text!r} is not a date. "
        f"Write it as YYYY-MM-DD (DD-MM-YYYY and DD/MM/YYYY are also read)."
    )
