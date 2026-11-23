"""Shared fixtures. Nothing here needs Ollama, a model or the network."""

from __future__ import annotations

import copy
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from docqa.config import Config, load_config

PROJECT_ROOT = Path(os.path.abspath(Path(__file__).parent.parent))


@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def config() -> Config:
    """The project's own config.yaml, loaded."""
    return load_config(PROJECT_ROOT / "config.yaml")


@pytest.fixture
def raw_config() -> dict[str, Any]:
    """The project's own config.yaml as plain data, for tests that change a value."""
    return yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))


@pytest.fixture
def write_config(tmp_path: Path):
    """Write a config dict to a temporary folder and return the file's path."""

    def _write(data: Any, folder: Path | None = None) -> Path:
        folder = folder or tmp_path
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "config.yaml"
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        return path

    return _write


TEST_RERANK_THRESHOLD = 0.3


@pytest.fixture
def temp_config(raw_config, write_config):
    """Build a configuration whose folders (docs, data, eval) all sit in a temporary folder.

    The project's own config.yaml is the starting point. The rerank threshold, which
    that file leaves unset on purpose, is given a value so that retrieval can run.
    Pass a function to change anything else before the file is written.
    """

    def _make(change: Callable[[dict[str, Any]], None] | None = None) -> Config:
        raw = copy.deepcopy(raw_config)
        raw["retrieval"]["rerank_score_threshold"] = TEST_RERANK_THRESHOLD
        if change is not None:
            change(raw)
        return load_config(write_config(raw))

    return _make


@pytest.fixture
def tmp_cfg(temp_config) -> Config:
    """A ready-made temporary configuration, for tests that need no special settings."""
    return temp_config()


@pytest.fixture
def store(tmp_cfg: Config):
    """An empty vector store in the temporary folder. Closed when the test ends."""
    from docqa.index.store import ChunkStore

    chunk_store = ChunkStore(tmp_cfg.qdrant)
    yield chunk_store
    chunk_store.close()


@pytest.fixture(autouse=True)
def _release_shared_resources():
    """No test may leave a shared model or an open store behind for the next one."""
    yield
    from docqa import resources

    resources.close_all()
