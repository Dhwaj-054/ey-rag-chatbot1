"""Load and validate ``config.yaml``.

Every model name, path, size and threshold the pipeline uses is read from that file.
The classes below define which keys exist and what values are allowed. They carry no
defaults for those settings, so a key missing from the file is an error rather than a
silent fallback.

Typical use::

    from docqa.config import load_config

    config = load_config()
    config.llm.answer_model      # "qwen2.5:7b"
    config.qdrant.path           # absolute Path to data/qdrant
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationError,
    ValidationInfo,
    model_validator,
)

CONFIG_ENV_VAR = "DOCQA_CONFIG"
CONFIG_FILE_NAME = "config.yaml"


class ConfigError(Exception):
    """The configuration file is missing, unreadable or invalid."""


def _anchor_to_config_dir(value: Path, info: ValidationInfo) -> Path:
    """Turn a relative path into an absolute one, based on the config file's folder."""
    base_dir = (info.context or {}).get("base_dir")
    if base_dir is None or value.is_absolute():
        return value
    return Path(os.path.abspath(base_dir / value))


ProjectPath = Annotated[Path, AfterValidator(_anchor_to_config_dir)]
NonEmptyStr = Annotated[str, Field(min_length=1)]


class _Section(BaseModel):
    """Base for all sections: unknown keys are rejected and values cannot be changed."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class PathsConfig(_Section):
    docs_dir: ProjectPath
    data_dir: ProjectPath
    parsed_dir: ProjectPath
    figures_dir: ProjectPath
    sqlite_path: ProjectPath
    manifest: ProjectPath
    eval_questions: ProjectPath
    eval_results_dir: ProjectPath


class LLMConfig(_Section):
    base_url: NonEmptyStr
    allow_remote: bool = False
    api_key_env: str | None = None
    vision_base_url: NonEmptyStr | None = None
    """Where the vision model is served, if that is not ``base_url``."""
    vision_api_key_env: str | None = None
    """The API key variable for ``vision_base_url``. Read only when that is set."""
    answer_model: NonEmptyStr
    vision_model: NonEmptyStr
    temperature: float = Field(ge=0.0, le=2.0)
    max_tokens: int = Field(gt=0)
    connect_timeout_s: float = Field(gt=0)
    request_timeout_s: float = Field(gt=0)

    @model_validator(mode="after")
    def _vision_key_needs_its_address(self) -> LLMConfig:
        if self.vision_api_key_env and not self.vision_base_url:
            raise ValueError(
                "vision_api_key_env is set but vision_base_url is not. Without vision_base_url "
                "the vision model uses base_url and api_key_env. Set vision_base_url, or "
                "remove vision_api_key_env"
            )
        return self

    def for_vision(self) -> LLMConfig:
        """These settings as the vision model's client needs them.

        They are the same settings unless ``vision_base_url`` names another server. Then
        that address and its own API key variable replace ``base_url`` and
        ``api_key_env``, so a key is never sent to a server it was not issued for.
        """
        if not self.vision_base_url:
            return self
        return self.model_copy(
            update={"base_url": self.vision_base_url, "api_key_env": self.vision_api_key_env}
        )


class ParsingConfig(_Section):
    artifacts_path: ProjectPath | None = None
    ocr: bool | Literal["auto"]
    """true: always. false: never. "auto": only for a document that has a page without
    a text layer."""
    scanned_page_max_chars: int = Field(ge=0)
    ocr_languages: list[str]
    images_scale: float = Field(gt=0)
    table_mode: Literal["accurate", "fast"]
    num_threads: int = Field(ge=0)
    """0 means one thread per physical core of this machine."""


class FiguresConfig(_Section):
    describe: bool
    min_page_fraction: float = Field(ge=0.0, lt=1.0)
    min_side_px: int = Field(ge=0)
    max_aspect_ratio: float = Field(ge=1.0)
    max_side_px: int = Field(gt=0)
    max_tokens: int = Field(gt=0)


class ScreeningConfig(_Section):
    enabled: bool
    flag_at: int = Field(gt=0)


class IngestionConfig(_Section):
    confirm_above_files: int = Field(gt=0)
    group_files: int = Field(gt=0)


class EmbeddingConfig(_Section):
    model: NonEmptyStr
    use_fp16: bool
    quantize: bool
    batch_size: int = Field(gt=0)
    max_length: int = Field(gt=0)


class RerankerConfig(_Section):
    model: NonEmptyStr
    use_fp16: bool
    quantize: bool
    batch_size: int = Field(gt=0)
    max_length: int = Field(gt=0)
    normalize: bool


class QdrantConfig(_Section):
    path: ProjectPath | None = None
    url: str | None = None
    collection: NonEmptyStr

    @model_validator(mode="after")
    def _exactly_one_location(self) -> QdrantConfig:
        if (self.path is None) == (self.url is None):
            raise ValueError("set exactly one of 'path' (local mode) or 'url' (server)")
        return self


class ChunkingConfig(_Section):
    min_tokens: int = Field(gt=0)
    max_tokens: int = Field(gt=0)
    overlap_min_tokens: int = Field(ge=0)
    overlap_max_tokens: int = Field(ge=0)

    @model_validator(mode="after")
    def _sizes_are_consistent(self) -> ChunkingConfig:
        if self.min_tokens > self.max_tokens:
            raise ValueError("min_tokens must not be larger than max_tokens")
        if self.overlap_min_tokens > self.overlap_max_tokens:
            raise ValueError("overlap_min_tokens must not be larger than overlap_max_tokens")
        if self.overlap_max_tokens >= self.min_tokens:
            raise ValueError("overlap_max_tokens must be smaller than min_tokens")
        return self


class RetrievalConfig(_Section):
    candidates: int = Field(gt=0)
    top_k: int = Field(gt=0)
    rrf_k: int = Field(gt=0)
    rerank_score_threshold: float | None = None
    default_access_group: NonEmptyStr

    @model_validator(mode="after")
    def _top_k_fits_in_candidates(self) -> RetrievalConfig:
        if self.top_k > self.candidates:
            raise ValueError("top_k must not be larger than candidates")
        return self


class GenerationConfig(_Section):
    not_found_message: NonEmptyStr
    evidence_char_budget: int = Field(gt=0)
    evidence_score_ratio: float = Field(ge=0.0, le=1.0)


class UIConfig(_Section):
    host: NonEmptyStr
    port: int = Field(gt=0, lt=65536)
    allow_network: bool = False
    warm_up: bool

    @model_validator(mode="after")
    def _local_unless_allowed(self) -> UIConfig:
        if not self.allow_network and self.host not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError(
                "host is not this machine. The chat screen would be reachable from the "
                "network. Use 127.0.0.1, or set allow_network to true on purpose"
            )
        return self


class Config(_Section):
    """The whole of ``config.yaml``, validated, with paths made absolute."""

    paths: PathsConfig
    llm: LLMConfig
    ui: UIConfig
    parsing: ParsingConfig
    figures: FiguresConfig
    screening: ScreeningConfig
    ingestion: IngestionConfig
    embedding: EmbeddingConfig
    reranker: RerankerConfig
    qdrant: QdrantConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig

    _config_file: Path | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _chunks_fit_the_models(self) -> Config:
        """A chunk longer than a model's input limit would be cut without any warning."""
        if self.embedding.max_length < self.chunking.max_tokens:
            raise ValueError(
                "embedding.max_length must not be smaller than chunking.max_tokens, "
                "or the end of long chunks would not be embedded"
            )
        if self.reranker.max_length < self.chunking.max_tokens:
            raise ValueError(
                "reranker.max_length must not be smaller than chunking.max_tokens, "
                "or the end of long chunks would not be read by the reranker"
            )
        return self

    @property
    def config_file(self) -> Path | None:
        """The file this configuration was loaded from (``None`` if built in code)."""
        return self._config_file

    @property
    def root_dir(self) -> Path | None:
        """The folder that holds the config file. Relative paths are based on it."""
        return self._config_file.parent if self._config_file else None


def find_config_file(path: str | os.PathLike[str] | None = None) -> Path:
    """Decide which config file to use.

    Order: the ``path`` argument, then the ``DOCQA_CONFIG`` environment variable, then
    ``config.yaml`` in the current folder, then ``config.yaml`` in the project root
    (the folder above ``src/``). A file named by the argument or the environment
    variable must exist; there is no fallback from those.
    """
    if path is not None:
        return _must_exist(Path(path), "the path given")
    from_env = os.environ.get(CONFIG_ENV_VAR)
    if from_env:
        return _must_exist(Path(from_env), f"the {CONFIG_ENV_VAR} environment variable")

    candidates = [
        Path.cwd() / CONFIG_FILE_NAME,
        Path(__file__).resolve().parents[2] / CONFIG_FILE_NAME,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    looked_in = ", ".join(str(candidate) for candidate in candidates)
    raise ConfigError(
        f"No {CONFIG_FILE_NAME} found. Looked in: {looked_in}. "
        f"Run from the project folder or set {CONFIG_ENV_VAR}."
    )


def _must_exist(path: Path, origin: str) -> Path:
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path} (from {origin}).")
    return path


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Read, validate and return the configuration.

    Relative paths in the file are resolved against the folder that holds it, so the
    result does not depend on the folder a command is run from.

    Raises:
        ConfigError: the file is missing, is not valid YAML, has an unknown or missing
            key, or holds a value that is not allowed. The message names the key.
    """
    config_file = Path(os.path.abspath(find_config_file(path)))
    try:
        raw: Any = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"Cannot read {config_file}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"{config_file} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{config_file} must contain a mapping of sections at the top level.")

    try:
        config = Config.model_validate(raw, context={"base_dir": config_file.parent})
    except ValidationError as exc:
        raise ConfigError(f"Invalid configuration in {config_file}:\n{_describe(exc)}") from exc
    config._config_file = config_file
    return config


def _describe(exc: ValidationError) -> str:
    """One line per problem, in the form ``section.key: what is wrong``."""
    lines = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "(top level)"
        lines.append(f"  {location}: {error['msg']}")
    return "\n".join(lines)
