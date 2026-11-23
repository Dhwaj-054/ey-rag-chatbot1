"""Write manifest.csv from the RBI document index.

    python scripts/build_manifest.py [INDEX.xlsx] [--out manifest.csv] [--docs docs]

The download script named each PDF after its title, cut to a fixed length, so the file
names are not the full titles. ``RBI_Documents_Index.xlsx`` holds, for every document,
its full title, its date and the file it was saved as. This script turns that into the
manifest that ingestion reads (``paths.manifest``), one row per file that exists:

    file            path below docs/, as the index gives it
    doc_title       the full title from the index
    version         the RBI reference number, where the index has one
    effective_date  the date in the index (the date RBI gives the document)
    access_group    left empty, which means the default group

Rows whose file is not in docs/ are left out and counted. Documents that are in docs/
but not in the index (anything added by hand) are not listed: ingestion gives those
their file name as their title. Add rows for them by hand if you want proper titles.

An existing manifest is not overwritten unless you pass --force.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import date, datetime
from pathlib import Path

import openpyxl

MANIFEST_COLUMNS = ("file", "doc_title", "version", "effective_date", "access_group")


def as_date(value: object) -> str:
    """A cell value as YYYY-MM-DD, or empty if it is not a date."""
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        for date_format in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d-%b-%Y"):
            try:
                return datetime.strptime(value.strip(), date_format).date().isoformat()
            except ValueError:
                continue
    return ""


def rows_from_index(index: Path) -> list[dict[str, str]]:
    """One manifest row per index row that names a local file."""
    book = openpyxl.load_workbook(index, read_only=True, data_only=True)
    rows: list[dict[str, str]] = []
    for sheet in book.worksheets:
        lines = sheet.iter_rows(values_only=True)
        header = [str(cell).strip() if cell is not None else "" for cell in next(lines, ())]
        if "Title" not in header or "Local File" not in header:
            continue  # the summary sheet
        column = {name: position for position, name in enumerate(header)}
        for line in lines:
            def cell(name: str, line: tuple = line) -> object:
                position = column.get(name)
                return line[position] if position is not None and position < len(line) else None

            local_file, title = cell("Local File"), cell("Title")
            if not local_file or not title:
                continue
            rows.append(
                {
                    "file": str(local_file).replace("\\", "/").strip(),
                    "doc_title": " ".join(str(title).split()),
                    "version": " ".join(str(cell("RBI Ref No.") or "").split()),
                    "effective_date": as_date(cell("Date")),
                    "access_group": "",
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Write manifest.csv from the RBI document index.")
    parser.add_argument("index", nargs="?", default="RBI_Documents_Index.xlsx", help="the index workbook")
    parser.add_argument("--out", default="manifest.csv", help="the manifest to write")
    parser.add_argument("--docs", default="docs", help="the folder that holds the documents")
    parser.add_argument("--force", action="store_true", help="overwrite an existing manifest")
    args = parser.parse_args()

    index, out, docs = Path(args.index), Path(args.out), Path(args.docs)
    if not index.is_file():
        print(f"The index workbook was not found: {index}")
        return 1
    if out.exists() and not args.force:
        print(f"{out} already exists. Pass --force to replace it.")
        return 1

    rows = rows_from_index(index)
    present: dict[str, dict[str, str]] = {}
    absent = duplicates = 0
    for row in rows:
        if not (docs / row["file"]).is_file():
            absent += 1
        elif row["file"].lower() in present:
            duplicates += 1
        else:
            present[row["file"].lower()] = row

    with open(out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(sorted(present.values(), key=lambda row: row["file"].lower()))

    print(f"{len(rows)} row(s) in the index.")
    print(f"{len(present)} written to {out}.")
    print(f"{absent} left out because the file is not in {docs}.")
    print(f"{duplicates} left out because the file was already listed.")
    without_date = sum(not row["effective_date"] for row in present.values())
    print(f"{without_date} written without a date.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
