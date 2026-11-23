"""The text version of a table: the form that is embedded and searched.

A table is stored twice. The original, as HTML, is what the chat screen shows. This
module writes the other form: plain text that starts with the table's title and its
column headers, then one line per row::

    Table: Schedule of charges for retail loans
    Columns: Loan type | Processing fee | Prepayment charge
    Home loan | 0.50% of the loan amount | Nil
    Personal loan | 2.00% of the loan amount | 3% of the outstanding principal

The title and headers are what make it findable. "Home loan | 0.50% | Nil" on its own
matches almost no question.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

UNTITLED = "(no title)"


def clean_cell(text: str) -> str:
    """One cell as a single line. A ``|`` inside a cell would read as a column break."""
    return re.sub(r"\s+", " ", text).replace("|", "/").strip()


def column_headers(rows: Sequence[Sequence[str]], header_rows: int) -> list[str]:
    """One header per column, from the first ``header_rows`` rows.

    Headers that span several rows are joined top-down with " / ", so a column under
    "Charges" and then "Processing fee" becomes "Charges / Processing fee". A header
    cell that spans several columns appears in each of them, which is what the join
    needs. A level that only repeats the one above it is left out.
    """
    width = max((len(row) for row in rows[:header_rows]), default=0)
    headers = []
    for column in range(width):
        parts: list[str] = []
        for row in rows[:header_rows]:
            cell = clean_cell(row[column]) if column < len(row) else ""
            if cell and (not parts or parts[-1] != cell):
                parts.append(cell)
        headers.append(" / ".join(parts))
    return headers


def table_to_text(title: str | None, rows: Sequence[Sequence[str]], header_rows: int) -> str:
    """Write a table as searchable text.

    Args:
        title: The table's caption, or failing that the heading it sits under.
        rows: Every row, each a list of cell texts. A cell that spans several rows or
            columns is expected to appear in each position it covers.
        header_rows: How many leading rows are column headers. With 0, the first row is
            taken as the headers when there is more than one row, because a table
            without headers is rare and a wrong guess costs little.
    """
    lines = [f"Table: {clean_cell(title) if title and title.strip() else UNTITLED}"]
    if header_rows == 0 and len(rows) > 1:
        header_rows = 1
    headers = column_headers(rows, header_rows)
    if any(headers):
        lines.append("Columns: " + " | ".join(headers))
    for row in rows[header_rows:]:
        cells = [clean_cell(cell) for cell in row]
        if any(cells):
            lines.append(" | ".join(cells))
    return "\n".join(lines)
