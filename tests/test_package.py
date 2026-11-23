"""The package as a whole: its privacy defaults and the stage interfaces."""

from __future__ import annotations

import importlib
import os
import tomllib
import typing

import pytest

import docqa

PRIVACY_DEFAULTS = {
    "HF_HUB_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "DO_NOT_TRACK": "1",
}

# The functions through which the stages connect. Phase 1 fills in their bodies.
STAGE_FUNCTIONS = {
    "docqa.ingest": [
        "file_sha256",
        "parse_document",
        "describe_figure",
        "chunk_elements",
        "ingest_path",
    ],
    "docqa.index": ["index_chunks", "remove_document", "count_chunks"],
    "docqa.retrieve": ["reciprocal_rank_fusion", "retrieve"],
    "docqa.generate": ["build_messages", "answer", "log_query"],
}


def test_importing_the_package_keeps_hugging_face_offline_and_silent(monkeypatch):
    for name in PRIVACY_DEFAULTS:
        monkeypatch.delenv(name, raising=False)

    importlib.reload(docqa)

    for name, value in PRIVACY_DEFAULTS.items():
        assert os.environ[name] == value, name


def test_a_value_already_in_the_environment_is_respected(monkeypatch):
    """Downloading an approved model is done by setting HF_HUB_OFFLINE=0 for one command."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")

    importlib.reload(docqa)

    assert os.environ["HF_HUB_OFFLINE"] == "0"


def test_the_project_installs_its_own_web_server_and_no_longer_streamlit(project_root):
    """The chat screen is served by Starlette and uvicorn (docqa.web)."""
    pinned = {
        line.split("==")[0].strip().lower()
        for line in (project_root / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if "==" in line and not line.lstrip().startswith("#")
    }

    assert {"starlette", "uvicorn"} <= pinned
    assert "streamlit" not in pinned
    project = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["tool"]["setuptools"]["package-data"]["docqa.web"] == ["static/*"]


@pytest.mark.parametrize(
    ("module_name", "function_name"),
    [(module, function) for module, functions in STAGE_FUNCTIONS.items() for function in functions],
)
def test_stage_function_is_declared_with_types_and_a_docstring(module_name, function_name):
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)

    assert function.__doc__, "missing docstring"
    hints = typing.get_type_hints(function)  # fails if a type hint names something undefined
    assert "return" in hints
