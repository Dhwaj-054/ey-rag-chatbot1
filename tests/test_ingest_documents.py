"""Document identity, the file hash, the manifest and the personal-data check."""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import pytest

from docqa.ingest.documents import (
    Manifest,
    ManifestError,
    clean_title,
    default_title,
    doc_id_for,
    document_info,
    file_sha256,
    source_path_for,
)
from docqa.ingest.screening import ScreeningResult, screen_elements, verhoeff_valid
from docqa.schema import Element, ElementType

# -- hash and identity ----------------------------------------------------------------


def test_file_sha256_matches_hashlib_and_changes_with_the_content(tmp_path):
    path = tmp_path / "a.pdf"
    path.write_bytes(b"x" * 3_000_000)
    assert file_sha256(path) == hashlib.sha256(b"x" * 3_000_000).hexdigest()

    path.write_bytes(b"x" * 2_999_999 + b"y")
    assert file_sha256(path) != hashlib.sha256(b"x" * 3_000_000).hexdigest()


def test_a_document_is_named_by_its_place_under_the_docs_folder(tmp_path):
    docs = tmp_path / "docs"
    inside = docs / "Master Circulars" / "Example.pdf"
    inside.parent.mkdir(parents=True)
    inside.write_bytes(b"pdf")
    outside = tmp_path / "elsewhere" / "Example.pdf"
    outside.parent.mkdir()
    outside.write_bytes(b"pdf")

    assert source_path_for(inside, docs) == "Master Circulars/Example.pdf"
    assert source_path_for(outside, docs) == outside.resolve().as_posix()


def test_doc_id_is_stable_short_and_ignores_case_and_slash_direction():
    assert doc_id_for("Master Circulars/Example.pdf") == doc_id_for("master circulars\\example.pdf")
    assert doc_id_for("a.pdf") != doc_id_for("b.pdf")
    assert len(doc_id_for("a.pdf")) == 16


def test_titles_are_made_safe_for_citations():
    assert clean_title("Reserve Bank of India [Agency Banks]\n Directions,  2026") == (
        "Reserve Bank of India (Agency Banks) Directions, 2026"
    )
    assert default_title(Path("tat-and-charges_of_standard__services.pdf")) == (
        "tat-and-charges of standard services"
    )


# -- manifest -------------------------------------------------------------------------


def write_manifest(tmp_path: Path, text: str, encoding: str = "utf-8") -> Path:
    path = tmp_path / "manifest.csv"
    path.write_text(text, encoding=encoding)
    return path


def test_a_missing_manifest_is_an_empty_one(tmp_path):
    manifest = Manifest.load(tmp_path / "manifest.csv")

    assert len(manifest) == 0
    assert manifest.lookup("anything.pdf") is None


def test_manifest_rows_are_found_by_path_whatever_the_case_or_slashes(tmp_path):
    manifest = Manifest.load(
        write_manifest(
            tmp_path,
            "file,doc_title,version,effective_date,access_group\n"
            "Master Circulars/Example.pdf,Master Circular - Example [2026],2026-27/12,2026-04-01,risk\n"
            "notice.pdf,,,,\n",
        )
    )

    entry = manifest.lookup("master circulars\\EXAMPLE.pdf")
    assert entry.doc_title == "Master Circular - Example (2026)"
    assert entry.version == "2026-27/12"
    assert entry.effective_date == date(2026, 4, 1)
    assert entry.access_group == "risk"
    # An empty cell means "use the default".
    bare = manifest.lookup("notice.pdf")
    assert (bare.doc_title, bare.version, bare.effective_date, bare.access_group) == (None,) * 4


def test_a_bare_file_name_finds_its_row_only_if_it_is_unique(tmp_path):
    manifest = Manifest.load(
        write_manifest(tmp_path, "file,doc_title\nA/one.pdf,First\nB/one.pdf,Second\nA/two.pdf,Third\n")
    )

    assert manifest.lookup("somewhere/else/two.pdf").doc_title == "Third"
    assert manifest.lookup("somewhere/else/one.pdf") is None


def test_manifest_written_by_excel_with_a_byte_order_mark_is_read(tmp_path):
    path = write_manifest(tmp_path, "file,doc_title\na.pdf,Title A\n", encoding="utf-8-sig")
    assert Manifest.load(path).lookup("a.pdf").doc_title == "Title A"


@pytest.mark.parametrize("written", ["2026-04-01", "01-04-2026", "01/04/2026"])
def test_dates_are_read_in_three_spellings(tmp_path, written):
    path = write_manifest(tmp_path, f"file,effective_date\na.pdf,{written}\n")
    assert Manifest.load(path).lookup("a.pdf").effective_date == date(2026, 4, 1)


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        pytest.param("doc_title\nTitle\n", "'file' column", id="no file column"),
        pytest.param("file,efective_date\na.pdf,2026-04-01\n", "efective_date", id="misspelt column"),
        pytest.param("file,doc_title\na.pdf,One\nA.PDF,Two\n", "listed twice", id="duplicate"),
        pytest.param("file,effective_date\na.pdf,April 2026\n", "row 2", id="bad date"),
        pytest.param("file,doc_title\n,Orphan title\n", "'file' is empty", id="empty file"),
    ],
)
def test_manifest_mistakes_are_named(tmp_path, content, expected):
    with pytest.raises(ManifestError, match=expected):
        Manifest.load(write_manifest(tmp_path, content))


def test_document_info_uses_the_manifest_and_falls_back_to_defaults(tmp_cfg):
    docs = tmp_cfg.paths.docs_dir
    docs.mkdir(parents=True)
    listed, unlisted = docs / "listed.pdf", docs / "sub" / "Notice_on_Media Policy.pdf"
    unlisted.parent.mkdir()
    listed.write_bytes(b"one")
    unlisted.write_bytes(b"two")
    manifest = Manifest.load(
        write_manifest(docs.parent, "file,doc_title,version,effective_date,access_group\n"
                       "listed.pdf,Listed Title,v7,2025-12-31,treasury\n")
    )

    first = document_info(listed, tmp_cfg, manifest)
    second = document_info(unlisted, tmp_cfg, manifest, sha256="given")

    assert (first.doc_title, first.version, first.effective_date, first.access_group) == (
        "Listed Title",
        "v7",
        date(2025, 12, 31),
        "treasury",
    )
    assert first.sha256 == hashlib.sha256(b"one").hexdigest()
    assert first.doc_id == doc_id_for("listed.pdf")
    assert (second.doc_title, second.version, second.effective_date, second.access_group) == (
        "Notice on Media Policy",
        None,
        None,
        "all",
    )
    assert second.sha256 == "given"
    assert second.doc_id == doc_id_for("sub/Notice_on_Media Policy.pdf")


# -- personal-data check --------------------------------------------------------------


def text_element(text: str, page: int = 1) -> Element:
    return Element("d", f"d:e{page}", page, None, ElementType.TEXT, text)


def aadhaar_like(first_eleven: str) -> str:
    """Complete eleven digits with their Verhoeff check digit, by trying all ten."""
    return next(first_eleven + str(d) for d in range(10) if verhoeff_valid(first_eleven + str(d)))


def test_verhoeff_check_digit():
    assert verhoeff_valid("2363")  # the textbook example: 236 has check digit 3
    assert not verhoeff_valid("2364")
    assert not verhoeff_valid("23a3")
    assert sum(verhoeff_valid("23456789012" + str(d)) for d in range(10)) == 1


def test_pan_like_strings_are_counted_once_each():
    found = screen_elements(
        [
            text_element("PAN ABCPE1234F and again ABCPE1234F, also AAACR5055K.", page=3),
            text_element("Not PANs: ABCXE1234F (wrong holder code), ABCPE12345, XABCPE1234F9.", page=4),
        ]
    )

    assert (found.pan_like, found.aadhaar_like, found.pages) == (2, 0, (3,))


def test_aadhaar_like_numbers_need_a_valid_check_digit():
    valid = aadhaar_like("23456789012")
    invalid = valid[:-1] + str((int(valid[-1]) + 1) % 10)
    grouped = f"{valid[:4]} {valid[4:8]} {valid[8:]}"

    found = screen_elements(
        [
            text_element(f"Aadhaar {grouped}.", page=2),
            text_element(f"Same number unspaced {valid}, and a wrong one {invalid}.", page=5),
            text_element("A long reference 1234567890123456 and a phone number 9876543210.", page=6),
        ]
    )

    assert (found.pan_like, found.aadhaar_like, found.pages) == (0, 1, (2, 5))


def test_a_file_is_flagged_from_the_configured_number_of_matches():
    result = ScreeningResult(pan_like=2, aadhaar_like=1, pages=(1, 4))

    assert result.total == 3
    assert result.flagged(3) and not result.flagged(4)


def test_the_report_gives_counts_and_pages_but_never_the_numbers_themselves():
    found = screen_elements([text_element("PAN ABCPE1234F.", page=9)])

    assert found.describe() == "1 PAN-like and 0 Aadhaar-like number(s), on page(s) 9"
    assert "ABCPE1234F" not in found.describe() and "ABCPE1234F" not in repr(found)
