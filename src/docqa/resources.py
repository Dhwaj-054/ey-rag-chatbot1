"""The heavy objects, loaded once per process and shared.

Loading BGE-M3 or the reranker takes many seconds, and Qdrant's local mode allows only
one open client per folder. So the stage functions (``index_chunks``, ``retrieve`` and
the others) do not create these objects themselves: they ask here, and get the same
object every time for the same configuration.

Each stage function also accepts the object as a keyword argument. Tests pass small
fakes that way, and never touch this module.

``close_all`` releases everything. Command-line programs call it before they exit, so
that the Qdrant folder is free for the next program.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from docqa.config import Config

if TYPE_CHECKING:
    from docqa.db import Database
    from docqa.index.embedder import Embedder
    from docqa.index.store import ChunkStore
    from docqa.llm import LLMClient
    from docqa.retrieve.reranker import Reranker
    from docqa.tokens import TokenCounter

_lock = threading.RLock()
_cache: dict[tuple[Any, ...], Any] = {}


def get_embedder(config: Config) -> Embedder:
    """The embedding model named in ``embedding``. Loaded on first use."""
    from docqa.index.embedder import BGEM3Embedder

    section = config.embedding
    key = ("embedder", section.model, section.use_fp16, section.batch_size, section.max_length)
    return _get(key, lambda: BGEM3Embedder(section))


def get_reranker(config: Config) -> Reranker:
    """The reranker named in ``reranker``. Loaded on first use."""
    from docqa.retrieve.reranker import BGEReranker

    section = config.reranker
    key = (
        "reranker",
        section.model,
        section.use_fp16,
        section.batch_size,
        section.max_length,
        section.normalize,
    )
    return _get(key, lambda: BGEReranker(section))


def get_store(config: Config) -> ChunkStore:
    """The vector store at ``qdrant.path`` or ``qdrant.url``. Opened on first use."""
    from docqa.index.store import ChunkStore

    section = config.qdrant
    key = ("store", str(section.path) if section.path else None, section.url, section.collection)
    return _get(key, lambda: ChunkStore(section))


def get_token_counter(config: Config) -> TokenCounter:
    """A counter that uses the embedding model's tokenizer. Loaded on first use."""
    from docqa.tokens import TokenizerCounter

    key = ("token_counter", config.embedding.model)
    return _get(key, lambda: TokenizerCounter(config.embedding.model))


def get_database(config: Config) -> Database:
    """The SQLite file at ``paths.sqlite_path``."""
    from docqa.db import Database

    key = ("database", str(config.paths.sqlite_path))
    return _get(key, lambda: Database(config.paths.sqlite_path))


def get_llm(config: Config) -> LLMClient:
    """The client for the model server in ``llm``. Creating it does not contact the server."""
    from docqa.llm import LLMClient

    key = ("llm", config.llm.model_dump_json())
    return _get(key, lambda: LLMClient(config.llm))


def close_all() -> None:
    """Release every shared object. Safe to call more than once."""
    with _lock:
        for key, value in list(_cache.items()):
            if key[0] in ("store", "llm"):
                try:
                    value.close()
                except Exception:  # noqa: BLE001 - closing must not hide the real error
                    pass
        _cache.clear()


def _get(key: tuple[Any, ...], create: Any) -> Any:
    with _lock:
        if key not in _cache:
            _cache[key] = create()
        return _cache[key]
