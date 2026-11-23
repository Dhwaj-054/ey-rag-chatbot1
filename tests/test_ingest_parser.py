"""The Docling wrapper: how its options are set, when OCR is used, and the parse cache.

Docling's models are not needed for any of this. The conversion itself is replaced by a
hand-built document. Converting a real PDF is covered by tests/test_real_models.py,
which needs the models.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
from docling.datamodel.base_models import ConversionStatus
from PIL import Image

from docqa.ingest import parser
from docqa.ingest.parser import (
    ParseError,
    build_pipeline_options,
    cache_paths,
    convert_pdf,
    get_converter,
    has_scanned_page,
    load_parsed,
    needs_ocr,
    parse_document,
    parse_settings,
    release_converters,
    thread_count,
)
from docqa.schema import DocumentInfo, ElementType
from tests.docling_samples import lending_policy
from tests.pdf_samples import scanned_pdf, text_pdf


def make_info(tmp_path: Path, sha256: str = "a" * 64, scanned: bool = False) -> DocumentInfo:
    source = tmp_path / "Lending Policy.pdf"
    if scanned:
        scanned_pdf(source, ["A SCANNED CIRCULAR"])
    else:
        text_pdf(source, ["Lending Policy", "The processing fee on a home loan is 0.50 percent."])
    return DocumentInfo("policy01", "Lending Policy", source, sha256, None, date(2026, 4, 1), "all")


def with_ocr(temp_config, value):
    def change(raw):
        raw["parsing"]["ocr"] = value

    return temp_config(change)


@pytest.fixture
def conversions(monkeypatch):
    """Replace Docling's conversion. Returns the list of (file name, ocr) it was asked for."""
    calls: list[tuple[str, bool]] = []

    def convert(path, config, ocr=True):
        calls.append((path.name, ocr))
        return lending_policy()

    monkeypatch.setattr(parser, "convert_pdf", convert)
    return calls


# -- options --------------------------------------------------------------------------


def test_docling_options_follow_the_configuration(temp_config, tmp_path):
    def change(raw):
        raw["parsing"].update(
            {
                "artifacts_path": str(tmp_path / "models" / "docling"),
                "ocr_languages": ["en"],
                "images_scale": 3.0,
                "table_mode": "fast",
                "num_threads": 2,
            }
        )

    options = build_pipeline_options(temp_config(change), ocr=False)

    assert Path(options.artifacts_path) == tmp_path / "models" / "docling"
    assert options.do_ocr is False
    assert options.ocr_options.lang == ["en"]
    assert options.images_scale == 3.0
    assert options.table_structure_options.mode.value == "fast"
    assert options.accelerator_options.num_threads == 2


def test_docling_options_that_this_project_depends_on(tmp_cfg):
    options = build_pipeline_options(tmp_cfg, ocr=True)

    # Without picture images there is nothing to show the vision model.
    assert options.generate_picture_images is True
    assert options.images_scale == 2.0
    # Nothing leaves the machine.
    assert options.enable_remote_services is False
    # Docling's own picture description is not used: every model call goes through docqa.llm.
    assert options.do_picture_description is False
    assert options.do_ocr is True and options.do_table_structure is True
    assert options.table_structure_options.mode.value == "accurate"
    assert Path(options.artifacts_path) == tmp_cfg.root_dir / "models" / "docling"


def test_without_an_artifacts_folder_docling_is_left_to_its_own_cache(temp_config):
    def unset(raw):
        raw["parsing"]["artifacts_path"] = None

    assert build_pipeline_options(temp_config(unset)).artifacts_path is None


def test_docling_gets_one_thread_per_physical_core_unless_a_number_is_set(tmp_cfg, temp_config):
    import psutil

    assert tmp_cfg.parsing.num_threads == 0
    assert thread_count(tmp_cfg) == (psutil.cpu_count(logical=False) or 1)
    assert build_pipeline_options(tmp_cfg).accelerator_options.num_threads == thread_count(tmp_cfg)

    def three(raw):
        raw["parsing"]["num_threads"] = 3

    assert thread_count(temp_config(three)) == 3


def test_one_converter_is_built_per_set_of_settings_and_can_be_released(tmp_cfg, temp_config):
    first = get_converter(tmp_cfg, ocr=True)

    assert get_converter(tmp_cfg, ocr=True) is first
    assert get_converter(tmp_cfg, ocr=False) is not first

    def other_scale(raw):
        raw["parsing"]["images_scale"] = 1.5

    assert get_converter(temp_config(other_scale), ocr=True) is not first

    release_converters()
    assert get_converter(tmp_cfg, ocr=True) is not first


# -- OCR only where it is needed ------------------------------------------------------


def test_a_pdf_with_a_text_layer_needs_no_ocr_and_a_scan_does(tmp_path):
    digital = text_pdf(tmp_path / "digital.pdf", ["Heading", "A sentence with a real text layer."])
    scan = scanned_pdf(tmp_path / "scan.pdf", ["A PICTURE OF TEXT"])

    assert has_scanned_page(digital, max_chars=20) is False
    assert has_scanned_page(scan, max_chars=20) is True


def test_a_page_without_text_or_picture_does_not_count_as_scanned(tmp_path):
    """An empty page has nothing for OCR to read."""
    almost_empty = text_pdf(tmp_path / "short.pdf", ["Hi"])

    assert has_scanned_page(almost_empty, max_chars=20) is False


def test_a_file_that_cannot_be_checked_keeps_ocr_on(tmp_path):
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"this is not a PDF")

    assert has_scanned_page(broken, max_chars=20) is True


def test_the_ocr_setting_decides_per_document(temp_config, tmp_path):
    digital = text_pdf(tmp_path / "digital.pdf", ["Heading", "A sentence with a real text layer."])
    scan = scanned_pdf(tmp_path / "scan.pdf", ["A PICTURE OF TEXT"])

    auto, always, never = (with_ocr(temp_config, value) for value in ("auto", True, False))

    assert (needs_ocr(digital, auto), needs_ocr(scan, auto)) == (False, True)
    assert (needs_ocr(digital, always), needs_ocr(scan, always)) == (True, True)
    assert (needs_ocr(digital, never), needs_ocr(scan, never)) == (False, False)


def test_parse_settings_describe_the_configuration_or_one_actual_parse(tmp_cfg):
    configured = parse_settings(tmp_cfg)
    actual = parse_settings(tmp_cfg, ocr=False)

    assert set(configured) == {"parser_version", "docling", "ocr", "ocr_languages", "images_scale", "table_mode"}
    assert configured["ocr"] == "auto" and actual["ocr"] is False
    assert configured["docling"] not in ("", "unknown")


# -- the parse cache ------------------------------------------------------------------


def test_a_document_is_parsed_once_and_then_read_from_the_cache(tmp_cfg, tmp_path, conversions):
    info = make_info(tmp_path)

    first = load_parsed(info, tmp_cfg)
    second = load_parsed(info, tmp_cfg)

    # The sample is a digital PDF, so with "auto" it is parsed without OCR.
    assert conversions == [("Lending Policy.pdf", False)]
    document_file, meta_file = cache_paths("policy01", tmp_cfg)
    assert document_file.is_file() and meta_file.is_file()
    assert document_file.parent == tmp_cfg.paths.parsed_dir
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    assert meta["sha256"] == "a" * 64 and meta["settings"] == parse_settings(tmp_cfg, ocr=False)
    assert len(list(second.iterate_items())) == len(list(first.iterate_items()))


def test_a_scanned_document_is_parsed_with_ocr(tmp_cfg, tmp_path, conversions):
    load_parsed(make_info(tmp_path, scanned=True), tmp_cfg)

    assert conversions == [("Lending Policy.pdf", True)]


def test_a_changed_file_or_a_changed_parse_setting_is_parsed_again(tmp_cfg, temp_config, tmp_path, conversions):
    load_parsed(make_info(tmp_path), tmp_cfg)

    load_parsed(make_info(tmp_path, sha256="b" * 64), tmp_cfg)
    assert len(conversions) == 2

    def bigger_images(raw):
        raw["parsing"]["images_scale"] = 3.0

    load_parsed(make_info(tmp_path, sha256="b" * 64), temp_config(bigger_images))
    assert len(conversions) == 3


def test_a_change_that_does_not_affect_parsing_uses_the_cache(tmp_cfg, temp_config, tmp_path, conversions):
    load_parsed(make_info(tmp_path), tmp_cfg)

    def smaller_chunks(raw):
        raw["chunking"]["max_tokens"] = 600

    load_parsed(make_info(tmp_path), temp_config(smaller_chunks))

    assert len(conversions) == 1


def test_a_parse_made_with_ocr_is_reused_where_ocr_is_no_longer_asked_for(temp_config, tmp_path, conversions):
    """It holds everything a parse without OCR would. Parsing again would only cost time."""
    info = make_info(tmp_path)
    load_parsed(info, with_ocr(temp_config, True))
    assert conversions == [("Lending Policy.pdf", True)]

    load_parsed(info, with_ocr(temp_config, "auto"))
    load_parsed(info, with_ocr(temp_config, False))

    assert len(conversions) == 1


def test_a_parse_made_without_ocr_is_not_reused_where_ocr_is_wanted(temp_config, tmp_path, conversions):
    info = make_info(tmp_path)
    load_parsed(info, with_ocr(temp_config, False))

    load_parsed(info, with_ocr(temp_config, True))

    assert conversions == [("Lending Policy.pdf", False), ("Lending Policy.pdf", True)]


def test_a_damaged_cache_file_is_replaced(tmp_cfg, tmp_path, conversions, caplog):
    info = make_info(tmp_path)
    load_parsed(info, tmp_cfg)
    cache_paths("policy01", tmp_cfg)[0].write_text("{ not json", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        document = load_parsed(info, tmp_cfg)

    assert len(conversions) == 2
    assert "Parsing again" in caplog.text
    assert len(list(document.iterate_items())) > 5


def test_parse_document_returns_elements_and_saves_the_figure_crops(tmp_cfg, tmp_path, conversions):
    elements = parse_document(make_info(tmp_path), tmp_cfg)

    kinds = [element.element_type for element in elements]
    assert kinds.count(ElementType.TABLE) == 1 and kinds.count(ElementType.FIGURE) == 1
    figure = next(element for element in elements if element.element_type is ElementType.FIGURE)
    assert figure.image_path.is_file()
    assert figure.image_path.parent == tmp_cfg.paths.figures_dir / "policy01"
    with Image.open(figure.image_path) as crop:
        assert crop.size == (600, 400)


# -- conversion results ---------------------------------------------------------------


class FakeConverter:
    def __init__(self, status=ConversionStatus.SUCCESS, errors=(), raises: Exception | None = None):
        self._result = SimpleNamespace(status=status, errors=list(errors), document=lending_policy())
        self._raises = raises
        self.calls: list[tuple] = []

    def convert(self, path, raises_on_error=True):
        self.calls.append((path, raises_on_error))
        if self._raises:
            raise self._raises
        return self._result


def use(monkeypatch, converter: FakeConverter) -> list[bool]:
    """Make ``get_converter`` return ``converter``, and record the OCR flag it is asked with."""
    asked: list[bool] = []

    def get(config, ocr=True):
        asked.append(ocr)
        return converter

    monkeypatch.setattr(parser, "get_converter", get)
    return asked


def test_a_successful_conversion_returns_docling_document(tmp_cfg, tmp_path, monkeypatch):
    converter = FakeConverter()
    asked = use(monkeypatch, converter)

    document = convert_pdf(tmp_path / "a.pdf", tmp_cfg, ocr=False)

    assert document.name == "lending-policy"
    assert converter.calls == [(tmp_path / "a.pdf", False)]
    assert asked == [False]


def test_a_failed_conversion_is_a_parse_error_with_doclings_reason(tmp_cfg, tmp_path, monkeypatch):
    use(monkeypatch, FakeConverter(ConversionStatus.FAILURE, [SimpleNamespace(error_message="page 3 is corrupt")]))
    with pytest.raises(ParseError, match="page 3 is corrupt"):
        convert_pdf(tmp_path / "a.pdf", tmp_cfg)

    use(monkeypatch, FakeConverter(raises=RuntimeError("layout model not found")))
    with pytest.raises(ParseError, match="RuntimeError: layout model not found"):
        convert_pdf(tmp_path / "a.pdf", tmp_cfg)


def test_a_partial_conversion_is_used_with_a_warning(tmp_cfg, tmp_path, monkeypatch, caplog):
    use(monkeypatch, FakeConverter(ConversionStatus.PARTIAL_SUCCESS, [SimpleNamespace(error_message="page 9 timed out")]))

    with caplog.at_level(logging.WARNING):
        document = convert_pdf(tmp_path / "a.pdf", tmp_cfg)

    assert document.name == "lending-policy"
    assert "parsed only in part: page 9 timed out" in caplog.text
