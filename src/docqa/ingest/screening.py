"""A check for personal identifiers, run on every document before it is indexed.

CLAUDE.md: the sample documents are public or sanitised, and a file that looks like it
holds real customer data must stop ingestion instead of being processed. This module is
that stop. It looks for two kinds of Indian identifier:

- PAN: ten characters, five letters, four digits, one letter, where the fourth letter
  is one of the ten holder-type codes;
- Aadhaar: twelve digits whose last digit is a valid Verhoeff check digit, written
  together or in groups of four.

A regulation may quote one or two such numbers to show the format, so a file is flagged
only from ``screening.flag_at`` distinct matches upward.

What it cannot see: account numbers (any run of digits could be one) and names next to
financial details. It is a tripwire for the obvious case, not proof that a file is
clean. The person who puts a file in ``docs/`` remains responsible for it.

The matched values are never returned, logged or printed: only how many there are and
on which pages.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from docqa.schema import Element

_PAN = re.compile(r"(?<![A-Z0-9])[A-Z]{3}[ABCFGHJLPT][A-Z][0-9]{4}[A-Z](?![A-Z0-9])")
_AADHAAR = re.compile(r"(?<![0-9])[2-9][0-9]{3}[ -]?[0-9]{4}[ -]?[0-9]{4}(?![0-9])")

# Verhoeff check-digit tables.
_MULTIPLY = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
_PERMUTE = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)


@dataclass(frozen=True)
class ScreeningResult:
    """What the check found in one document. It holds counts and pages, never values."""

    pan_like: int
    """Distinct PAN-like strings."""
    aadhaar_like: int
    """Distinct Aadhaar-like numbers with a valid check digit."""
    pages: tuple[int, ...]
    """Pages on which at least one was found, in order."""

    @property
    def total(self) -> int:
        return self.pan_like + self.aadhaar_like

    def flagged(self, flag_at: int) -> bool:
        return self.total >= flag_at

    def describe(self) -> str:
        pages = ", ".join(str(page) for page in self.pages[:20])
        more = " and others" if len(self.pages) > 20 else ""
        return (
            f"{self.pan_like} PAN-like and {self.aadhaar_like} Aadhaar-like number(s), "
            f"on page(s) {pages}{more}"
        )


def verhoeff_valid(digits: str) -> bool:
    """True if the last digit of ``digits`` is the Verhoeff check digit of the rest."""
    if not digits.isdigit():
        return False
    check = 0
    for position, digit in enumerate(reversed(digits)):
        check = _MULTIPLY[check][_PERMUTE[position % 8][int(digit)]]
    return check == 0


def screen_elements(elements: Iterable[Element]) -> ScreeningResult:
    """Count personal identifiers in the text of a parsed document."""
    pans: set[str] = set()
    aadhaars: set[str] = set()
    pages: set[int] = set()
    for element in elements:
        if not element.text:
            continue
        found_here = False
        for match in _PAN.finditer(element.text):
            pans.add(match.group())
            found_here = True
        for match in _AADHAAR.finditer(element.text):
            number = re.sub(r"[ -]", "", match.group())
            if verhoeff_valid(number):
                aadhaars.add(number)
                found_here = True
        if found_here:
            pages.add(element.page)
    return ScreeningResult(len(pans), len(aadhaars), tuple(sorted(pages)))
