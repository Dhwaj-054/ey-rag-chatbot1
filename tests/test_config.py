"""Tests for the config loader."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from docqa.config import CONFIG_ENV_VAR, ConfigError, find_config_file, load_config


@pytest.fixture(autouse=True)
def _no_config_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests must not be steered by a DOCQA_CONFIG left in the developer's shell."""
    monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)


# -- the project's own config.yaml ----------------------------------------------------


def test_project_config_holds_the_values_decided_in_claude_md(config):
    """If one of these is changed on purpose, change CLAUDE.md and this test with it."""
    assert config.llm.base_url == "http://localhost:11434/v1"
    assert config.llm.allow_remote is False
    assert config.llm.answer_model == "qwen2.5:7b"
    assert config.llm.vision_model == "qwen2.5vl:7b"

    assert config.embedding.model == "BAAI/bge-m3"
    assert config.embedding.use_fp16 is False
    assert config.reranker.model == "BAAI/bge-reranker-v2-m3"
    assert config.reranker.use_fp16 is False

    assert (config.chunking.min_tokens, config.chunking.max_tokens) == (400, 800)
    assert (config.chunking.overlap_min_tokens, config.chunking.overlap_max_tokens) == (50, 100)

    assert config.retrieval.candidates == 15
    assert config.retrieval.top_k == 4
    assert config.retrieval.default_access_group == "all"

    assert config.qdrant.url is None
    assert config.generation.not_found_message == "I could not find this in the documents."

    # Docling's built-in OCR for scanned pages ("auto": only for documents that have
    # one), and picture crops for the vision model.
    assert config.parsing.ocr == "auto"
    assert config.figures.describe is True
    # The 8-bit option. Off for the embedding model: switching it means indexing every
    # document again. On for the reranker since 2026-11-23: it is the same model, with
    # its weights held as 8-bit numbers, and ranks about twice as fast.
    assert config.embedding.quantize is False and config.reranker.quantize is True
    # A file that looks like real customer data stops ingestion.
    assert config.screening.enabled is True
    # A temporary value, set on the laptop that runs the models for its first documents.
    # It is to be chosen again from real test questions (evaluate --suggest-threshold).
    assert config.retrieval.rerank_score_threshold == 0.5


def test_project_config_does_not_cut_chunks_before_the_models_see_them(config):
    """FlagEmbedding's own limit is 512 tokens for both models; chunks go up to 800."""
    assert config.embedding.max_length >= config.chunking.max_tokens
    assert config.reranker.max_length >= config.chunking.max_tokens
    assert config.paths.manifest == config.root_dir / "manifest.csv"
    # Docling's models sit in a git-ignored folder inside the project.
    assert config.parsing.artifacts_path == config.root_dir / "models" / "docling"


def test_paths_are_absolute_and_based_on_the_config_folder(config, project_root):
    assert config.config_file == project_root / "config.yaml"
    assert config.root_dir == project_root

    assert config.qdrant.path == project_root / "data" / "qdrant"
    assert config.paths.docs_dir == project_root / "docs"
    assert config.paths.data_dir == project_root / "data"
    assert config.paths.sqlite_path == project_root / "data" / "docqa.sqlite"
    assert config.paths.eval_questions == project_root / "eval" / "questions.csv"
    for name in type(config.paths).model_fields:
        assert getattr(config.paths, name).is_absolute(), name


def test_paths_do_not_depend_on_the_current_folder(tmp_path, monkeypatch, project_root):
    monkeypatch.chdir(tmp_path)
    config = load_config(project_root / "config.yaml")
    assert config.qdrant.path == project_root / "data" / "qdrant"


def test_relative_paths_follow_a_config_file_kept_elsewhere(tmp_path, raw_config, write_config):
    path = write_config(raw_config, tmp_path / "elsewhere")
    config = load_config(path)
    assert config.root_dir == tmp_path / "elsewhere"
    assert config.qdrant.path == tmp_path / "elsewhere" / "data" / "qdrant"
    assert config.paths.docs_dir == tmp_path / "elsewhere" / "docs"


def test_absolute_paths_are_kept_as_given(tmp_path, raw_config, write_config):
    outside = tmp_path / "outside" / "qdrant"
    raw_config["qdrant"]["path"] = str(outside)
    config = load_config(write_config(raw_config, tmp_path / "project"))
    assert config.qdrant.path == outside


def test_config_cannot_be_changed_after_loading(config):
    with pytest.raises(ValidationError):
        config.llm.answer_model = "something-else"


# -- which file is used ---------------------------------------------------------------


def test_environment_variable_selects_the_file(monkeypatch, raw_config, write_config):
    raw_config["llm"]["answer_model"] = "model-from-env-file"
    monkeypatch.setenv(CONFIG_ENV_VAR, str(write_config(raw_config)))
    assert load_config().llm.answer_model == "model-from-env-file"


def test_path_argument_wins_over_environment_variable(
    tmp_path, monkeypatch, raw_config, write_config
):
    raw_config["llm"]["answer_model"] = "model-from-env-file"
    monkeypatch.setenv(CONFIG_ENV_VAR, str(write_config(raw_config, tmp_path / "env")))
    raw_config["llm"]["answer_model"] = "model-from-argument"
    explicit = write_config(raw_config, tmp_path / "arg")
    assert load_config(explicit).llm.answer_model == "model-from-argument"


def test_config_in_the_current_folder_is_found(tmp_path, monkeypatch, raw_config, write_config):
    path = write_config(raw_config)
    monkeypatch.chdir(tmp_path)
    assert find_config_file() == path


def test_falls_back_to_the_project_config(tmp_path, monkeypatch, project_root):
    monkeypatch.chdir(tmp_path)  # no config.yaml here
    assert find_config_file() == project_root / "config.yaml"


def test_missing_file_named_by_argument_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_missing_file_named_by_environment_variable_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv(CONFIG_ENV_VAR, str(tmp_path / "nope.yaml"))
    with pytest.raises(ConfigError, match=CONFIG_ENV_VAR):
        load_config()


# -- what is rejected -----------------------------------------------------------------


def test_invalid_yaml_is_an_error(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("llm: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(path)


@pytest.mark.parametrize("content", ["", "- a\n- b\n", "just a string\n"])
def test_file_must_hold_sections(tmp_path, content):
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(path)


def _set(section: str, key: str, value):
    def change(raw):
        raw[section][key] = value

    return change


def _drop(section: str, key: str | None = None):
    def change(raw):
        if key is None:
            del raw[section]
        else:
            del raw[section][key]

    return change


def _qdrant(path, url):
    def change(raw):
        raw["qdrant"]["path"] = path
        raw["qdrant"]["url"] = url

    return change


@pytest.mark.parametrize(
    ("change", "expected_in_message"),
    [
        pytest.param(_set("llm", "answer_modle", "x"), "llm.answer_modle", id="misspelt key"),
        pytest.param(_set("llm", "answer_model", ""), "llm.answer_model", id="empty model name"),
        pytest.param(_set("llm", "temperature", -1), "llm.temperature", id="negative temperature"),
        pytest.param(_set("llm", "max_tokens", 0), "llm.max_tokens", id="zero max_tokens"),
        pytest.param(_set("embedding", "batch_size", 0), "embedding.batch_size", id="zero batch"),
        pytest.param(_set("retrieval", "candidates", "many"), "retrieval.candidates", id="type"),
        pytest.param(_set("retrieval", "top_k", 31), "top_k must not be larger", id="top_k"),
        pytest.param(_set("chunking", "min_tokens", 900), "min_tokens must not be", id="min>max"),
        pytest.param(_set("chunking", "overlap_min_tokens", 200), "overlap_min", id="overlap order"),
        pytest.param(_set("chunking", "overlap_max_tokens", 400), "smaller than", id="overlap size"),
        pytest.param(_drop("retrieval", "top_k"), "retrieval.top_k", id="missing key"),
        pytest.param(_drop("chunking"), "chunking", id="missing section"),
        pytest.param(_drop("parsing"), "parsing", id="missing parsing section"),
        pytest.param(_set("parsing", "table_mode", "best"), "parsing.table_mode", id="table mode"),
        pytest.param(_set("parsing", "images_scale", 0), "parsing.images_scale", id="zero scale"),
        pytest.param(_set("parsing", "ocr_languages", "en"), "parsing.ocr_languages", id="not a list"),
        pytest.param(_set("figures", "min_page_fraction", 1.5), "figures.min_page_fraction", id="frac"),
        pytest.param(_set("screening", "flag_at", 0), "screening.flag_at", id="flag at zero"),
        pytest.param(_set("ingestion", "confirm_above_files", 0), "ingestion.confirm", id="confirm"),
        pytest.param(_drop("reranker", "normalize"), "reranker.normalize", id="normalize missing"),
        pytest.param(_set("embedding", "max_length", 512), "embedding.max_length must", id="embed cut"),
        pytest.param(_set("reranker", "max_length", 512), "reranker.max_length must", id="rerank cut"),
        pytest.param(_set("generation", "evidence_char_budget", 0), "evidence_char_budget", id="budget"),
        pytest.param(_drop("paths", "manifest"), "paths.manifest", id="manifest missing"),
        pytest.param(_set("parsing", "ocr", "sometimes"), "parsing.ocr", id="ocr word"),
        pytest.param(_set("parsing", "num_threads", -1), "parsing.num_threads", id="threads"),
        pytest.param(_set("figures", "max_aspect_ratio", 0.5), "figures.max_aspect_ratio", id="aspect"),
        pytest.param(_set("figures", "max_side_px", 0), "figures.max_side_px", id="side"),
        pytest.param(_set("ingestion", "group_files", 0), "ingestion.group_files", id="group"),
        pytest.param(_drop("embedding", "quantize"), "embedding.quantize", id="quantize missing"),
        pytest.param(_set("generation", "evidence_score_ratio", 1.5), "evidence_score_ratio", id="ratio"),
        pytest.param(_drop("ui"), "ui", id="missing ui section"),
        pytest.param(_set("ui", "port", 70000), "ui.port", id="port"),
        pytest.param(_set("ui", "host", "0.0.0.0"), "reachable from the network", id="open host"),
        pytest.param(_qdrant("data/qdrant", "http://localhost:6333"), "exactly one", id="both"),
        pytest.param(_qdrant(None, None), "exactly one", id="neither"),
    ],
)
def test_invalid_settings_are_rejected_and_named(
    raw_config, write_config, change, expected_in_message
):
    change(raw_config)
    with pytest.raises(ConfigError) as error:
        load_config(write_config(raw_config))
    assert expected_in_message in str(error.value)


def test_unknown_section_is_rejected(raw_config, write_config):
    raw_config["surprise"] = {"key": 1}
    with pytest.raises(ConfigError, match="surprise"):
        load_config(write_config(raw_config))


# -- what is allowed ------------------------------------------------------------------


def test_rerank_threshold_may_be_unset_or_a_number(raw_config, write_config):
    raw_config["retrieval"]["rerank_score_threshold"] = None
    assert load_config(write_config(raw_config)).retrieval.rerank_score_threshold is None
    raw_config["retrieval"]["rerank_score_threshold"] = 0.35
    assert load_config(write_config(raw_config)).retrieval.rerank_score_threshold == 0.35


@pytest.mark.parametrize(("written", "read"), [(True, True), (False, False), ("auto", "auto")])
def test_ocr_may_be_on_off_or_automatic(raw_config, write_config, written, read):
    raw_config["parsing"]["ocr"] = written
    assert load_config(write_config(raw_config)).parsing.ocr == read


def test_qdrant_server_instead_of_local_folder(raw_config, write_config):
    """The later move to a Qdrant server is a config change."""
    raw_config["qdrant"]["path"] = None
    raw_config["qdrant"]["url"] = "http://qdrant.internal:6333"
    config = load_config(write_config(raw_config))
    assert config.qdrant.path is None
    assert config.qdrant.url == "http://qdrant.internal:6333"


def test_paths_section_accepts_path_objects_in_code(config):
    """Paths are real Path objects, ready to use."""
    assert isinstance(config.paths.docs_dir, Path)


# -- where the models are served ---------------------------------------------------------


def test_the_vision_model_is_served_with_the_answer_model_unless_told_otherwise(config):
    assert config.llm.vision_base_url is None
    assert config.llm.for_vision() is config.llm


def test_the_vision_model_can_have_its_own_server_and_its_own_key(temp_config):
    """Answers from a hosted endpoint, figures still described on this machine."""

    def hosted_answers(raw):
        raw["llm"].update(
            base_url="https://models.example.com/v1",
            allow_remote=True,
            api_key_env="HOSTED_KEY",
            vision_base_url="http://localhost:11434/v1",
        )

    llm = temp_config(hosted_answers).llm
    vision = llm.for_vision()

    assert (llm.base_url, llm.api_key_env) == ("https://models.example.com/v1", "HOSTED_KEY")
    assert vision.base_url == "http://localhost:11434/v1"
    assert vision.api_key_env is None  # the hosted service's key is not sent to another server
    assert vision.vision_model == llm.vision_model


def test_a_vision_key_without_a_vision_address_is_a_mistake(temp_config):
    def key_only(raw):
        raw["llm"]["vision_api_key_env"] = "VISION_KEY"

    with pytest.raises(ConfigError, match="vision_base_url"):
        temp_config(key_only)


# -- settings of one machine: config.local.yaml -------------------------------------------------


def test_a_local_file_is_laid_over_the_configuration(raw_config, write_config, tmp_path):
    """What differs on one machine is kept out of the file that is in git."""
    path = write_config(raw_config)
    documents = tmp_path / "documents on another drive"
    (tmp_path / "config.local.yaml").write_text(
        f"paths:\n  docs_dir: {documents.as_posix()!r}\nretrieval:\n  candidates: 8\n  top_k: 3\n",
        encoding="utf-8",
    )

    config = load_config(path)
    plain = load_config(path, use_local=False)

    assert config.paths.docs_dir == documents
    assert (config.retrieval.candidates, config.retrieval.top_k) == (8, 3)
    # Everything the local file does not name keeps its value, in the same section too.
    assert config.retrieval.rrf_k == raw_config["retrieval"]["rrf_k"]
    assert config.paths.uploads_dir == tmp_path / "data" / "uploads"
    assert config.llm.answer_model == "qwen2.5:7b"
    assert config.local_file == tmp_path / "config.local.yaml"
    assert plain.paths.docs_dir == tmp_path / "docs" and plain.local_file is None


def test_without_a_local_file_nothing_changes_and_an_empty_one_is_fine(raw_config, write_config, tmp_path):
    path = write_config(raw_config)
    assert load_config(path).local_file is None

    (tmp_path / "config.local.yaml").write_text("# nothing differs on this machine yet\n", encoding="utf-8")
    config = load_config(path)

    assert config.local_file is not None and config.retrieval.candidates == raw_config["retrieval"]["candidates"]


def test_a_mistake_in_the_local_file_is_reported_as_one(raw_config, write_config, tmp_path):
    path = write_config(raw_config)
    local = tmp_path / "config.local.yaml"

    local.write_text("retrieval:\n  candidats: 8\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"config\.local\.yaml laid over it.*\n.*retrieval\.candidats"):
        load_config(path)

    local.write_text("- not\n- a mapping\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"config\.local\.yaml must contain a mapping"):
        load_config(path)

    local.write_text("retrieval: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="is not valid YAML"):
        load_config(path)


def test_nothing_limits_the_time_of_an_answer_or_of_the_ranking(raw_config, write_config):
    """Limits were tried on 2026-11-23 and taken out: they cost evidence, and so answers."""
    assert "time_limit_s" not in raw_config["generation"]
    assert "rerank_time_limit_s" not in raw_config["retrieval"]

    raw_config["generation"]["time_limit_s"] = 120
    with pytest.raises(ConfigError, match=r"generation\.time_limit_s"):
        load_config(write_config(raw_config))


def test_the_largest_upload_must_be_a_size(temp_config):
    def no_size(raw):
        raw["ui"]["max_upload_mb"] = 0

    with pytest.raises(ConfigError, match=r"ui\.max_upload_mb"):
        temp_config(no_size)


def test_the_projects_own_settings_are_tested_without_a_local_file(project_root, config):
    """A config.local.yaml on the machine that runs the tests must not change what they check."""
    assert config.local_file is None
    assert "config.local.yaml" in (project_root / ".gitignore").read_text(encoding="utf-8")


# -- another answer model -------------------------------------------------------------------


def test_nothing_about_thinking_is_sent_unless_the_setting_says_so(config, temp_config):
    assert config.llm.reasoning_effort is None  # right for qwen2.5:7b, which does not think

    def another_model(raw):
        raw["llm"].update(answer_model="qwen3:4b", reasoning_effort="none")

    llm = temp_config(another_model).llm
    assert (llm.answer_model, llm.reasoning_effort) == ("qwen3:4b", "none")

    def empty(raw):
        raw["llm"]["reasoning_effort"] = ""

    with pytest.raises(ConfigError, match=r"llm\.reasoning_effort"):
        temp_config(empty)
