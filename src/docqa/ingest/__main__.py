"""Command line for ingestion.

    python -m docqa.ingest [PATH] [options]

PATH is a PDF or a folder of PDFs. It defaults to ``paths.docs_dir``. Run with
``--help`` for the options. The exit status is 0 when every file was ingested or
skipped, 1 when at least one failed, and 2 when the run stopped to ask for ``--yes``.

Stop the chat app before running this. Both use the same index, and Qdrant's local
mode lets only one program open it.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from docqa import resources
from docqa.config import ConfigError, load_config
from docqa.index.embedder import EmbeddingError
from docqa.index.store import StoreError
from docqa.ingest.documents import ManifestError
from docqa.ingest.pipeline import IngestAborted, ingest_path
from docqa.tokens import TokenCounterError


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m docqa.ingest",
        description="Parse, describe, chunk and index the documents for the Q&A chatbot.",
    )
    parser.add_argument("path", nargs="?", help="a PDF or a folder of PDFs (default: paths.docs_dir)")
    parser.add_argument("--config", help="path to config.yaml (default: the project's own)")
    parser.add_argument("--limit", type=int, metavar="N", help="ingest at most N files in this run")
    parser.add_argument("--force", action="store_true", help="ingest files even if unchanged")
    parser.add_argument("--yes", action="store_true", help="do not stop to ask when many files need work")
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="do not call the vision model; figures are cropped but not described",
    )
    parser.add_argument(
        "--allow-flagged",
        action="store_true",
        help="ingest files that the personal-data check flags (look at the file first)",
    )
    parser.add_argument(
        "--prune", action="store_true", help="also remove documents whose files are gone"
    )
    parser.add_argument("--list", action="store_true", help="list the ingested documents and exit")
    parser.add_argument("-v", "--verbose", action="store_true", help="show library messages too")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be 1 or more")

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(exc)
        return 1

    try:
        if args.list:
            return _list_documents(config)
        started = time.perf_counter()
        report = ingest_path(
            Path(args.path) if args.path else config.paths.docs_dir,
            config,
            limit=args.limit,
            force=args.force,
            assume_yes=args.yes,
            allow_flagged=args.allow_flagged,
            describe_figures=False if args.no_figures else None,
            prune=args.prune,
            progress=print,
        )
    except IngestAborted as exc:
        print(exc)
        return 2
    except (StoreError, EmbeddingError, TokenCounterError, ManifestError, FileNotFoundError) as exc:
        print(f"Ingestion could not start: {exc}")
        return 1
    finally:
        resources.close_all()

    print()
    print(
        f"Done in {time.perf_counter() - started:.0f} s. Seen {report.files_seen}, "
        f"unchanged {report.files_skipped}, ingested {report.files_ingested}, "
        f"failed {len(report.failures)}. Chunks indexed: {report.chunks_indexed}. "
        f"Figures described by the vision model: {report.figures_described}."
    )
    for title, lines in (("Warnings", report.warnings), ("Failures", report.failures)):
        if lines:
            print(f"\n{title}:")
            for line in lines:
                print(f"  - {line}")
    return 1 if report.failures else 0


def _list_documents(config) -> int:
    records = resources.get_database(config).list_documents()
    if not records:
        print("No documents have been ingested yet.")
        return 0
    for record in records:
        date = record.effective_date.isoformat() if record.effective_date else "no date"
        print(
            f"{record.doc_title}\n    {record.source_path}\n    {record.pages} page(s), "
            f"{record.chunks} chunk(s), effective {date}, group {record.access_group}, "
            f"ingested {record.ingested_at}"
        )
    print(f"\n{len(records)} document(s), {sum(record.chunks for record in records)} chunk(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
