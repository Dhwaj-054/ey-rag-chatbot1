"""Check that this machine is ready to run the pipeline.

    python -m docqa.check_env [--config PATH]

Prints PASS or FAIL for each requirement and exits with status 1 if anything failed.

It only looks. Nothing is downloaded: models are checked through the LLM server's
model list and the local Hugging Face cache. The one thing it writes is a throwaway
test collection in the configured Qdrant location, which it deletes again.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import importlib.metadata
import shutil
import sys
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from docqa.config import Config, ConfigError, LLMConfig, load_config
from docqa.llm import LLMClient, LLMError, is_local_url, model_is_listed

MIN_PYTHON = (3, 11)

# (name on PyPI, module that has to import)
REQUIRED_PACKAGES = (
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("docling", "docling.document_converter"),
    ("FlagEmbedding", "FlagEmbedding"),
    ("huggingface_hub", "huggingface_hub"),
    ("qdrant-client", "qdrant_client"),
    ("httpx", "httpx"),
    ("PyYAML", "yaml"),
    ("pydantic", "pydantic"),
    ("starlette", "starlette"),
    ("uvicorn", "uvicorn"),
    ("pypdfium2", "pypdfium2"),
    ("pillow", "PIL"),
    ("psutil", "psutil"),
    ("pytest", "pytest"),
)

# A model counts as present when one of these weight files is there.
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


@dataclass(frozen=True)
class CheckResult:
    """The outcome of one check."""

    name: str
    passed: bool
    detail: str

    def __str__(self) -> str:
        return f"[{'PASS' if self.passed else 'FAIL'}] {self.name}: {self.detail}"


def check_python() -> CheckResult:
    """The interpreter is new enough."""
    found = ".".join(str(part) for part in sys.version_info[:3])
    needed = ".".join(str(part) for part in MIN_PYTHON)
    return CheckResult(
        "Python version", sys.version_info >= MIN_PYTHON, f"{found} ({needed} or newer needed)"
    )


def check_packages() -> list[CheckResult]:
    """Every required package imports. One result per package."""
    results = []
    for package, module in REQUIRED_PACKAGES:
        name = f"Package imports: {package}"
        try:
            importlib.import_module(module)
            detail = importlib.metadata.version(package)
            if package == "torch":
                detail += _torch_build()
        except Exception as exc:  # an import can fail in many ways, e.g. a DLL that will not load
            results.append(CheckResult(name, False, _summary(exc)))
        else:
            results.append(CheckResult(name, True, detail))
    return results


def check_llm(config: Config, client: LLMClient | None = None) -> list[CheckResult]:
    """The LLM server answers, and it has the answer model and the vision model.

    When ``llm.vision_base_url`` names a second server, each server is checked for its
    own model. ``client`` is for tests; it then stands in for every server.
    """
    answer = ("answer", config.llm.answer_model)
    vision = ("vision", config.llm.vision_model)
    vision_settings = config.llm.for_vision()
    if vision_settings.base_url.rstrip("/") == config.llm.base_url.rstrip("/"):
        return _check_llm_server(config.llm, (answer, vision), client)
    return [
        *_check_llm_server(config.llm, (answer,), client),
        *_check_llm_server(vision_settings, (vision,), client),
    ]


def remote_notes(config: Config) -> list[str]:
    """One sentence for each kind of content that these settings send off this machine."""
    notes = []
    if not is_local_url(config.llm.base_url):
        notes.append(
            f"the answer model runs on {urlsplit(config.llm.base_url).hostname}, not on this "
            f"machine. Every question, and the passages found for it, are sent there."
        )
    vision_url = config.llm.for_vision().base_url
    if not is_local_url(vision_url):
        notes.append(
            f"the vision model runs on {urlsplit(vision_url).hostname}, not on this machine. "
            f"During ingestion every figure image is sent there (--no-figures sends none)."
        )
    return notes


def setup_notes(config: Config) -> list[str]:
    """Things about this machine's settings that are worth knowing and are not failures."""
    notes = []
    if config.local_file is not None:
        notes.append(
            f"settings of this machine are taken from {config.local_file.name}, which is laid "
            f"over config.yaml."
        )
    docs = config.paths.docs_dir
    pdfs = sum(1 for item in docs.rglob("*") if item.suffix.lower() == ".pdf") if docs.is_dir() else 0
    if pdfs:
        notes.append(f"the documents folder holds {pdfs} PDF file(s): {docs}")
    else:
        notes.append(
            f"the documents folder holds no PDF files: {docs}. If the documents are kept "
            f"somewhere else on this machine, name that folder in config.local.yaml:\n"
            f"    paths:\n      docs_dir: \"D:/where/the/documents/are\""
        )
    limit = config.generation.time_limit_s
    if limit is not None:
        notes.append(
            f"an answer is planned to take at most {limit:g} s (generation.time_limit_s). "
            f"'python -m docqa.stats' shows whether the answers given so far kept to it."
        )
    return notes


def _check_llm_server(
    settings: LLMConfig, wanted: Sequence[tuple[str, str]], client: LLMClient | None
) -> list[CheckResult]:
    server = f"LLM server reachable ({settings.base_url})"
    try:
        if client is None:
            with LLMClient(settings) as own_client:
                available = own_client.list_models()
        else:
            available = client.list_models()
    except LLMError as exc:
        results = [CheckResult(server, False, str(exc))]
        for role, model in wanted:
            results.append(
                CheckResult(
                    f"Model present: {model} ({role})",
                    False,
                    "not checked, because the server is not reachable",
                )
            )
        return results

    results = [CheckResult(server, True, f"{len(available)} model(s) on the server")]
    for role, model in wanted:
        present = model_is_listed(model, available)
        results.append(
            CheckResult(
                f"Model present: {model} ({role})",
                present,
                "on the server" if present else "not on the server",
            )
        )
    return results


def check_local_models(config: Config) -> list[CheckResult]:
    """The embedding model and the reranker are on disk. Looks only; never downloads."""
    return [
        _local_model("embedding", config.embedding.model),
        _local_model("reranker", config.reranker.model),
    ]


def check_docling_models(config: Config) -> CheckResult:
    """Docling's own models (layout, tables, OCR) are in the folder the configuration names.

    Docling fetches its models from Hugging Face the first time it parses a document.
    This project keeps Hugging Face offline, so they have to be downloaded once, on
    purpose, into a folder that ``parsing.artifacts_path`` points at.
    """
    name = "Docling models (parsing.artifacts_path)"
    folder = config.parsing.artifacts_path
    if folder is None:
        return CheckResult(
            name,
            False,
            "not set, so Docling would look for its models online. Download them once with "
            "'docling-tools models download layout tableformer rapidocr -o models/docling', "
            "then set parsing.artifacts_path to models/docling.",
        )
    how = (
        f"Download them once with 'docling-tools models download layout tableformer rapidocr "
        f'-o "{folder}"\'.'
    )
    if not folder.is_dir():
        return CheckResult(name, False, f"{folder} does not exist. {how}")
    found = sorted(item.name for item in folder.iterdir() if item.is_dir())
    if not found:
        return CheckResult(name, False, f"{folder} holds no model folders. {how}")
    return CheckResult(name, True, f"{len(found)} model folder(s) in {folder}: {', '.join(found)}")


def check_qdrant(config: Config) -> CheckResult:
    """A test collection can be created and deleted at the configured Qdrant location."""
    location = str(config.qdrant.path) if config.qdrant.path else str(config.qdrant.url)
    name = f"Qdrant create and delete ({location})"
    collection = f"check_env_{uuid.uuid4().hex[:12]}"
    step = "load qdrant-client"
    client = None
    deleted = False
    failure: str | None = None
    try:
        from qdrant_client import QdrantClient, models

        step = "open the store"
        if config.qdrant.path:
            client = QdrantClient(path=str(config.qdrant.path))
        else:
            client = QdrantClient(url=config.qdrant.url)
        step = f"create test collection {collection}"
        client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE),
        )
        if not client.collection_exists(collection):
            failure = "the test collection was not there after creating it"
        else:
            step = f"delete test collection {collection}"
            client.delete_collection(collection_name=collection)
            deleted = True
            if client.collection_exists(collection):
                failure = f"test collection {collection} is still there after deleting it"
    except Exception as exc:  # any failure here is a failed check, whatever its type
        failure = f"could not {step}: {_summary(exc)}"
    finally:
        if client is not None:
            client.close()

    if deleted:
        leftover = _remove_collection_files(config, collection)
        if leftover is not None and failure is None:
            failure = f"the test collection was deleted, but its files are still there: {leftover}"
    if failure is not None:
        return CheckResult(name, False, failure)
    return CheckResult(name, True, "test collection created and deleted")


def run_checks(config: Config) -> list[CheckResult]:
    """Run every check that needs the configuration."""
    return [
        *check_llm(config),
        *check_local_models(config),
        check_docling_models(config),
        check_qdrant(config),
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m docqa.check_env",
        description="Check that this machine is ready to run the document Q&A pipeline.",
    )
    parser.add_argument("--config", help="path to config.yaml (default: the project's own)")
    args = parser.parse_args(argv)

    # A message with an unusual character must not crash the report on a Windows console.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    results: list[CheckResult] = []

    def report(new_results: Sequence[CheckResult]) -> None:
        for result in new_results:
            print(result, flush=True)
        results.extend(new_results)

    report([check_python()])
    report(check_packages())
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        report([CheckResult("Configuration", False, str(exc))])
        print("\nThe remaining checks need the configuration and were skipped.")
    else:
        report([CheckResult("Configuration", True, str(config.config_file))])
        report(check_llm(config))
        report(check_local_models(config))
        report([check_docling_models(config)])
        report([check_qdrant(config)])
        for note in (*remote_notes(config), *setup_notes(config)):
            print(f"\nNote: {note}")
        if config.retrieval.rerank_score_threshold is None:
            print(
                "\nNote: retrieval.rerank_score_threshold is not set yet. That is expected "
                "before the first evaluation run; the chatbot cannot answer until it is."
            )

    failed = sum(not result.passed for result in results)
    print(f"\n{len(results) - failed} passed, {failed} failed")
    return 1 if failed else 0


def _remove_collection_files(config: Config, collection: str) -> Path | None:
    """Remove what qdrant-client leaves on disk for a deleted local collection.

    On Windows, local mode (seen with qdrant-client 1.20.0) keeps the deleted
    collection's ``storage.sqlite`` open until Python's garbage collector has run, so
    its own attempt to remove the folder fails silently. If the file stays, a later
    collection with the same name gets the old points back. Collecting garbage first
    releases the file.

    Returns the folder if it is still there afterwards, otherwise ``None``.
    """
    if not config.qdrant.path:
        return None
    folder = Path(config.qdrant.path) / "collection" / collection
    if folder.exists():
        gc.collect()
        shutil.rmtree(folder, ignore_errors=True)
    return folder if folder.exists() else None


def _local_model(role: str, model: str) -> CheckResult:
    name = f"Model present: {model} ({role})"
    folder = Path(model)
    if folder.is_dir():
        found = any((folder / filename).is_file() for filename in WEIGHT_FILES)
        return CheckResult(
            name, found, "weights found in that folder" if found else "no weights in that folder"
        )
    try:
        from huggingface_hub import try_to_load_from_cache

        found = any(
            isinstance(try_to_load_from_cache(model, filename), str) for filename in WEIGHT_FILES
        )
    except Exception as exc:  # a failed lookup is a failed check, whatever its type
        return CheckResult(name, False, f"could not look in the local cache: {_summary(exc)}")
    return CheckResult(
        name,
        found,
        "found in the local Hugging Face cache" if found else "not in the local Hugging Face cache",
    )


def _torch_build() -> str:
    import torch

    return ", CUDA available" if torch.cuda.is_available() else ", CPU only"


def _summary(exc: BaseException) -> str:
    """The exception's type and the first line of its message."""
    lines = str(exc).strip().splitlines()
    return f"{type(exc).__name__}: {lines[0]}" if lines else type(exc).__name__


if __name__ == "__main__":
    raise SystemExit(main())
