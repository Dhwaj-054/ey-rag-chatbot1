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


def test_streamlit_serves_this_machine_only_and_sends_no_usage_statistics(project_root):
    """Seen when the app was first started: without ``server.address`` Streamlit listens
    on the whole network and, when headless, looks up the public IP address online."""
    settings = tomllib.loads(
        (project_root / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    )

    assert settings["server"]["address"] == "localhost"
    assert settings["browser"]["gatherUsageStats"] is False


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
