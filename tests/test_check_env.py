"""Tests for the environment check.

The LLM server is faked, the Hugging Face cache is a temporary folder, and Qdrant runs
in a temporary folder, so these tests need no model and no network.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from docqa import check_env
from docqa.check_env import (
    CheckResult,
    check_docling_models,
    check_llm,
    check_local_models,
    check_packages,
    check_python,
    check_qdrant,
    main,
)
from docqa.config import Config
from docqa.llm import LLMClient


def client_with_models(config: Config, *models: str) -> LLMClient:
    body = {"object": "list", "data": [{"id": model} for model in models]}
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=body))
    return LLMClient(config.llm, transport=transport)


def client_that_cannot_connect(config: Config) -> LLMClient:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    return LLMClient(config.llm, transport=httpx.MockTransport(refuse))


def with_qdrant_path(config: Config, path: Path) -> Config:
    return config.model_copy(update={"qdrant": config.qdrant.model_copy(update={"path": path})})


def with_model_names(config: Config, embedding: str, reranker: str) -> Config:
    return config.model_copy(
        update={
            "embedding": config.embedding.model_copy(update={"model": embedding}),
            "reranker": config.reranker.model_copy(update={"model": reranker}),
        }
    )


def add_to_fake_hub_cache(cache: Path, repo_id: str, filename: str) -> None:
    """Lay out one cached file the way huggingface_hub stores it."""
    repo = cache / ("models--" + repo_id.replace("/", "--"))
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text("0123abcd", encoding="utf-8")
    snapshot = repo / "snapshots" / "0123abcd"
    snapshot.mkdir(parents=True)
    (snapshot / filename).write_bytes(b"weights")


# -- Python and packages --------------------------------------------------------------


def test_python_version_passes_on_the_interpreter_running_the_tests():
    result = check_python()
    assert result.passed
    assert result.name == "Python version"


def test_packages_are_reported_one_by_one(monkeypatch):
    monkeypatch.setattr(
        check_env,
        "REQUIRED_PACKAGES",
        (("pytest", "pytest"), ("no-such-package", "no_such_package_for_docqa")),
    )

    present, missing = check_packages()

    assert present.passed and present.name == "Package imports: pytest"
    assert not missing.passed and missing.name == "Package imports: no-such-package"
    assert "ModuleNotFoundError" in missing.detail


# -- LLM server and its models --------------------------------------------------------


def test_llm_checks_pass_when_the_server_has_both_models(config):
    client = client_with_models(config, config.llm.answer_model, config.llm.vision_model, "x:1")

    server, answer, vision = check_llm(config, client)

    assert server.passed and config.llm.base_url in server.name
    assert answer.passed and config.llm.answer_model in answer.name
    assert vision.passed and config.llm.vision_model in vision.name


def test_llm_check_names_the_model_that_is_missing(config):
    client = client_with_models(config, config.llm.answer_model)

    server, answer, vision = check_llm(config, client)

    assert server.passed
    assert answer.passed
    assert not vision.passed
    assert config.llm.vision_model in vision.name
    assert vision.detail == "not on the server"


def test_llm_checks_fail_when_the_server_is_not_running(config):
    server, answer, vision = check_llm(config, client_that_cannot_connect(config))

    assert not server.passed and "Cannot reach the LLM server" in server.detail
    assert not answer.passed and "not checked" in answer.detail
    assert not vision.passed and "not checked" in vision.detail


# -- embedding model and reranker -----------------------------------------------------


def test_local_models_are_found_in_the_hugging_face_cache(config, tmp_path, monkeypatch):
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))
    add_to_fake_hub_cache(tmp_path, config.embedding.model, "pytorch_model.bin")

    embedding, reranker = check_local_models(config)

    assert embedding.passed and config.embedding.model in embedding.name
    assert not reranker.passed and config.reranker.model in reranker.name
    assert reranker.detail == "not in the local Hugging Face cache"


def test_local_models_can_be_plain_folders(config, tmp_path):
    with_weights = tmp_path / "bge-m3"
    with_weights.mkdir()
    (with_weights / "model.safetensors").write_bytes(b"weights")
    without_weights = tmp_path / "empty"
    without_weights.mkdir()

    embedding, reranker = check_local_models(
        with_model_names(config, str(with_weights), str(without_weights))
    )

    assert embedding.passed
    assert not reranker.passed and reranker.detail == "no weights in that folder"


# -- Qdrant ---------------------------------------------------------------------------


def test_qdrant_check_creates_and_removes_its_test_collection(config, tmp_path):
    from qdrant_client import QdrantClient

    store = tmp_path / "qdrant"

    result = check_qdrant(with_qdrant_path(config, store))

    assert result.passed, result.detail
    # Nothing is left on disk either. qdrant-client alone leaves the deleted collection's
    # file behind on Windows, and a later collection of that name would inherit its data.
    assert list((store / "collection").iterdir()) == []
    client = QdrantClient(path=str(store))
    try:
        assert client.get_collections().collections == []
    finally:
        client.close()


def test_qdrant_check_fails_when_the_folder_cannot_be_used(config, tmp_path):
    blocker = tmp_path / "a-file"
    blocker.write_text("not a folder", encoding="utf-8")

    result = check_qdrant(with_qdrant_path(config, blocker / "qdrant"))

    assert not result.passed
    assert "could not open the store" in result.detail


# -- the report -----------------------------------------------------------------------


def test_result_line_format():
    assert str(CheckResult("Python version", True, "3.14.6")) == "[PASS] Python version: 3.14.6"
    assert str(CheckResult("Qdrant", False, "locked")) == "[FAIL] Qdrant: locked"


@pytest.fixture
def canned_checks(monkeypatch):
    """Replace the slow, machine-dependent checks so that main() itself can be tested."""

    def install(*, qdrant_ok: bool) -> None:
        monkeypatch.setattr(check_env, "check_packages", lambda: [CheckResult("pkg", True, "1.0")])
        monkeypatch.setattr(check_env, "check_llm", lambda config: [CheckResult("llm", True, "up")])
        monkeypatch.setattr(check_env, "check_local_models", lambda config: [])
        monkeypatch.setattr(
            check_env, "check_docling_models", lambda config: CheckResult("docling", True, "there")
        )
        monkeypatch.setattr(
            check_env, "check_qdrant", lambda config: CheckResult("qdrant", qdrant_ok, "detail")
        )

    return install


def test_main_exits_zero_when_everything_passes(canned_checks, project_root, capsys):
    canned_checks(qdrant_ok=True)

    status = main(["--config", str(project_root / "config.yaml")])

    output = capsys.readouterr().out
    assert status == 0
    assert "[PASS] Configuration" in output
    assert "[FAIL]" not in output
    assert "6 passed, 0 failed" in output
    # The project's config.yaml leaves the threshold unset on purpose; that is a note, not a failure.
    assert "rerank_score_threshold is not set yet" in output


def test_main_exits_one_when_a_check_fails(canned_checks, project_root, capsys):
    canned_checks(qdrant_ok=False)

    status = main(["--config", str(project_root / "config.yaml")])

    output = capsys.readouterr().out
    assert status == 1
    assert "[FAIL] qdrant: detail" in output
    assert "5 passed, 1 failed" in output


def test_main_says_nothing_about_the_threshold_once_it_is_set(canned_checks, tmp_cfg, capsys):
    canned_checks(qdrant_ok=True)

    assert main(["--config", str(tmp_cfg.config_file)]) == 0
    assert "rerank_score_threshold" not in capsys.readouterr().out


# -- Docling's own models -------------------------------------------------------------


def test_docling_models_must_be_in_a_configured_folder(temp_config, tmp_path):
    def with_folder(folder):
        def change(raw):
            raw["parsing"]["artifacts_path"] = str(folder) if folder else None

        return temp_config(change)

    not_set = check_docling_models(with_folder(None))
    assert not not_set.passed
    assert "not set" in not_set.detail and "docling-tools models download" in not_set.detail

    missing = check_docling_models(with_folder(tmp_path / "models" / "docling"))
    assert not missing.passed and "does not exist" in missing.detail

    folder = tmp_path / "models" / "docling"
    folder.mkdir(parents=True)
    empty = check_docling_models(with_folder(folder))
    assert not empty.passed and "holds no model folders" in empty.detail

    (folder / "docling-project--docling-layout-heron").mkdir()
    (folder / "RapidOcr").mkdir()
    present = check_docling_models(with_folder(folder))
    assert present.passed
    assert "2 model folder(s)" in present.detail and "RapidOcr" in present.detail


def test_main_reports_a_broken_configuration_and_skips_what_needs_it(
    canned_checks, tmp_path, capsys
):
    canned_checks(qdrant_ok=True)

    status = main(["--config", str(tmp_path / "missing.yaml")])

    output = capsys.readouterr().out
    assert status == 1
    assert "[FAIL] Configuration" in output
    assert "[PASS] llm" not in output
