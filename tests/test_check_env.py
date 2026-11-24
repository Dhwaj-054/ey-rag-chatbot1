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
    remote_notes,
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


def with_llm(config: Config, **changes) -> Config:
    return config.model_copy(update={"llm": config.llm.model_copy(update=changes)})


def test_each_server_is_checked_for_its_own_model(config):
    """vLLM serves one model per server, so the vision model can have its own address."""
    config = with_llm(config, vision_base_url="http://localhost:9000/v1")
    client = client_with_models(config, config.llm.answer_model)

    answer_server, answer, vision_server, vision = check_llm(config, client)

    assert config.llm.base_url in answer_server.name and answer.passed
    assert "http://localhost:9000/v1" in vision_server.name and vision_server.passed
    assert not vision.passed and config.llm.vision_model in vision.name


def test_what_leaves_the_machine_is_said_in_plain_words(config):
    assert remote_notes(config) == []  # the project's own settings keep everything here

    answers_hosted = with_llm(
        config,
        base_url="https://models.example.com/v1",
        allow_remote=True,
        vision_base_url="http://localhost:11434/v1",
    )
    everything_hosted = with_llm(config, base_url="https://models.example.com/v1", allow_remote=True)

    (only_note,) = remote_notes(answers_hosted)
    assert "answer model runs on models.example.com" in only_note
    assert "Every question" in only_note
    answer_note, vision_note = remote_notes(everything_hosted)
    assert "answer model" in answer_note
    assert "vision model runs on models.example.com" in vision_note and "figure image" in vision_note


def test_every_package_the_check_looks_for_is_pinned_in_requirements(project_root):
    def normal(name: str) -> str:
        return name.lower().replace("_", "-")

    pinned = {
        normal(line.split("==")[0].strip())
        for line in (project_root / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if "==" in line and not line.lstrip().startswith("#")
    }

    looked_for = {normal(package) for package, _module in check_env.REQUIRED_PACKAGES}

    assert looked_for - pinned == set()
    assert "streamlit" not in looked_for


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
    # Everything is on this machine and the threshold is set, so nothing is flagged. The
    # one note that is left says where the documents are.
    assert "not on this machine" not in output and "rerank_score_threshold is not set" not in output
    assert "the documents folder holds" in output


def test_main_notes_a_threshold_that_is_not_set_without_failing(canned_checks, temp_config, capsys):
    def unset(raw):
        raw["retrieval"]["rerank_score_threshold"] = None

    canned_checks(qdrant_ok=True)

    assert main(["--config", str(temp_config(unset).config_file)]) == 0
    assert "rerank_score_threshold is not set yet" in capsys.readouterr().out


def test_main_says_what_a_hosted_endpoint_is_sent(canned_checks, temp_config, capsys):
    def hosted(raw):
        raw["llm"].update(base_url="https://models.example.com/v1", allow_remote=True)

    canned_checks(qdrant_ok=True)

    assert main(["--config", str(temp_config(hosted).config_file)]) == 0
    output = capsys.readouterr().out
    assert "Note: the answer model runs on models.example.com" in output
    assert "Note: the vision model runs on models.example.com" in output


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


# -- notes about this machine's settings ----------------------------------------------------


def test_the_notes_say_where_the_documents_are(tmp_cfg):
    from docqa.check_env import setup_notes

    empty = setup_notes(tmp_cfg)
    tmp_cfg.paths.docs_dir.mkdir(parents=True)
    (tmp_cfg.paths.docs_dir / "Circulars").mkdir()
    (tmp_cfg.paths.docs_dir / "Circulars" / "a.PDF").write_bytes(b"%PDF-1.7")
    (tmp_cfg.paths.docs_dir / "notes.txt").write_text("not a document", encoding="utf-8")
    filled = setup_notes(tmp_cfg)

    assert any("holds no PDF files" in note and "config.local.yaml" in note for note in empty)
    assert any("holds 1 PDF file(s)" in note for note in filled)
    assert not any("config.local.yaml, which is laid over" in note for note in filled)


def test_the_notes_say_when_a_local_file_is_in_use(raw_config, write_config, tmp_path):
    from docqa.check_env import setup_notes
    from docqa.config import load_config

    path = write_config(raw_config)
    (tmp_path / "config.local.yaml").write_text("retrieval:\n  candidates: 8\n  top_k: 3\n", encoding="utf-8")

    notes = setup_notes(load_config(path))

    assert any("taken from config.local.yaml" in note for note in notes)
