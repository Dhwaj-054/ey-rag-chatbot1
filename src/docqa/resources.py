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

import gc
import threading
import time
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
_key_locks: dict[tuple[Any, ...], threading.Lock] = {}


def get_embedder(config: Config) -> Embedder:
    """The embedding model named in ``embedding``. Loaded on first use."""
    from docqa.index.embedder import BGEM3Embedder

    section = config.embedding
    key = ("embedder", section.model_dump_json())
    return _get(key, lambda: BGEM3Embedder(section))


def get_reranker(config: Config) -> Reranker:
    """The reranker named in ``reranker``. Loaded on first use."""
    from docqa.retrieve.reranker import BGEReranker

    section = config.reranker
    key = ("reranker", section.model_dump_json())
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


def warm_up(config: Config) -> dict[str, float]:
    """Load everything a question needs, so that the first question does not wait for it.

    Opens the vector store and loads the embedding model and the reranker. Returns the
    seconds each one took. An error (a model that is not on this machine, a store that
    another program has open) is passed on to the caller.
    """
    seconds: dict[str, float] = {}
    for name, load in (("store", get_store), ("embedder", get_embedder), ("reranker", get_reranker)):
        started = time.perf_counter()
        load(config)
        seconds[name] = time.perf_counter() - started
    return seconds


def release(kind: str) -> None:
    """Forget the shared objects of one kind (``"embedder"``, ``"reranker"``, ...).

    Ingestion uses this between its steps, so that a model it no longer needs does not
    take up memory next to the one it needs now.
    """
    with _lock:
        for key in [key for key in _cache if key[0] == kind]:
            value = _cache.pop(key)
            if kind in ("store", "llm"):
                try:
                    value.close()
                except Exception:  # noqa: BLE001 - closing must not hide the real error
                    pass
    gc.collect()


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
    """Return the object for ``key``, creating it once.

    Each key has its own lock. Loading a model takes a long time, and it must not make
    a request for something else (the database, say) wait behind it.
    """
    with _lock:
        if key in _cache:
            return _cache[key]
        key_lock = _key_locks.setdefault(key, threading.Lock())
    with key_lock:
        with _lock:
            if key in _cache:
                return _cache[key]
        value = create()
        with _lock:
            _cache[key] = value
        return value
