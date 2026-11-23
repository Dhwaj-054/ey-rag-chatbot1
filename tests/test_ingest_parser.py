"""The Docling wrapper: how its options are set, and the parse cache.

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

from docqa.ingest import parser
from docqa.ingest.parser import (
    ParseError,
    build_pipeline_options,
    cache_paths,
    convert_pdf,
    get_converter,
    load_parsed,
    parse_document,
    parse_settings,
)
from docqa.schema import DocumentInfo, ElementType
from tests.docling_samples import lending_policy


def make_info(tmp_path: Path, sha256: str = "a" * 64) -> DocumentInfo:
    source = tmp_path / "Lending Policy.pdf"
    source.write_bytes(b"%PDF-1.7")
    return DocumentInfo("policy01", "Lending Policy", source, sha256, None, date(2026, 4, 1), "all")


@pytest.fixture
def conversions(monkeypatch):
    calls: list[str] = []

    def convert(path, config):
        calls.append(path.name)
        return lending_policy()

    monkeypatch.setattr(parser, "convert_pdf", convert)
    return calls


# -- options --------------------------------------------------------------------------


def test_docling_options_follow_the_configuration(temp_config, tmp_path):
    def change(raw):
        raw["parsing"].update(
            {
                "artifacts_path": str(tmp_path / "models" / "docling"),
                "ocr": False,
                "ocr_languages": ["en"],
                "images_scale": 3.0,
                "table_mode": "fast",
                "num_threads": 2,
            }
        )

    options = build_pipeline_options(temp_config(change))

    assert Path(options.artifacts_path) == tmp_path / "models" / "docling"
    assert options.do_ocr is False
    assert options.ocr_options.lang == ["en"]
    assert options.images_scale == 3.0
    assert options.table_structure_options.mode.value == "fast"
    assert options.accelerator_options.num_threads == 2


def test_docling_options_that_this_project_depends_on(tmp_cfg):
    options = build_pipeline_options(tmp_cfg)

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


def test_one_converter_is_built_per_set_of_settings(tmp_cfg, temp_config):
    first = get_converter(tmp_cfg)

    assert get_converter(tmp_cfg) is first

    def other_scale(raw):
        raw["parsing"]["images_scale"] = 1.5

    assert get_converter(temp_config(other_scale)) is not first


def test_parse_settings_name_everything_that_changes_a_parse(tmp_cfg):
    settings = parse_settings(tmp_cfg)

    assert set(settings) == {"parser_version", "docling", "ocr", "ocr_languages", "images_scale", "table_mode"}
    assert settings["docling"] not in ("", "unknown")


# -- the parse cache ------------------------------------------------------------------


def test_a_document_is_parsed_once_and_then_read_from_the_cache(tmp_cfg, tmp_path, conversions):
    info = make_info(tmp_path)

    first = load_parsed(info, tmp_cfg)
    second = load_parsed(info, tmp_cfg)

    assert conversions == ["Lending Policy.pdf"]
    document_file, meta_file = cache_paths("policy01", tmp_cfg)
    assert document_file.is_file() and meta_file.is_file()
    assert document_file.parent == tmp_cfg.paths.parsed_dir
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    assert meta["sha256"] == "a" * 64 and meta["settings"] == parse_settings(tmp_cfg)
    assert len(list(second.iterate_items())) == len(list(first.iterate_items()))


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


def test_a_successful_conversion_returns_docling_document(tmp_cfg, tmp_path, monkeypatch):
    converter = FakeConverter()
    monkeypatch.setattr(parser, "get_converter", lambda config: converter)

    document = convert_pdf(tmp_path / "a.pdf", tmp_cfg)

    assert document.name == "lending-policy"
    assert converter.calls == [(tmp_path / "a.pdf", False)]


def test_a_failed_conversion_is_a_parse_error_with_doclings_reason(tmp_cfg, tmp_path, monkeypatch):
    failed = FakeConverter(ConversionStatus.FAILURE, [SimpleNamespace(error_message="page 3 is corrupt")])
    monkeypatch.setattr(parser, "get_converter", lambda config: failed)
    with pytest.raises(ParseError, match="page 3 is corrupt"):
        convert_pdf(tmp_path / "a.pdf", tmp_cfg)

    crashed = FakeConverter(raises=RuntimeError("layout model not found"))
    monkeypatch.setattr(parser, "get_converter", lambda config: crashed)
    with pytest.raises(ParseError, match="RuntimeError: layout model not found"):
        convert_pdf(tmp_path / "a.pdf", tmp_cfg)


def test_a_partial_conversion_is_used_with_a_warning(tmp_cfg, tmp_path, monkeypatch, caplog):
    partial = FakeConverter(ConversionStatus.PARTIAL_SUCCESS, [SimpleNamespace(error_message="page 9 timed out")])
    monkeypatch.setattr(parser, "get_converter", lambda config: partial)

    with caplog.at_level(logging.WARNING):
        document = convert_pdf(tmp_path / "a.pdf", tmp_cfg)

    assert document.name == "lending-policy"
    assert "parsed only in part: page 9 timed out" in caplog.text
