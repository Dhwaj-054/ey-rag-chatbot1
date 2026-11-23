"""Stage 2, indexing: embed chunks and store them where retrieval can find them.

The store is Qdrant, at the location given in the ``qdrant`` section of the
configuration (local mode today). ``docqa.index.store`` lists what to know about local
mode, including why a collection is never dropped and re-created here.

Each function takes the embedding model and the store as optional keyword arguments.
Left out, the shared ones from ``docqa.resources`` are used.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from docqa.config import Config
from docqa.index.embedder import Embedder
from docqa.index.store import ChunkStore
from docqa.schema import Chunk

log = logging.getLogger(__name__)

# Chunks are embedded and stored a group at a time, so that a long run shows progress
# and an interrupted one has stored what it finished. Inside a group the embedding
# model works in batches of ``embedding.batch_size``.
_GROUP_SIZE = 32


def searchable_text(chunk: Chunk) -> str:
    """The text that is embedded and reranked for a chunk.

    It is the chunk's text with the document title and the headings above it in front.
    A chunk cut from the middle of a section does not contain the section's heading,
    and without it a passage such as "The limit is 25 per cent" cannot be told apart
    from the same sentence in another section. The title does the same job across
    documents: the sample set holds the same direction once per type of bank, and
    only the title says which one a passage belongs to.
    """
    lines = [chunk.doc_title]
    if chunk.heading_path:
        lines.append(" > ".join(chunk.heading_path))
    lines.append(chunk.text)
    return "\n".join(lines)


def index_chunks(
    chunks: Sequence[Chunk],
    config: Config,
    *,
    embedder: Embedder | None = None,
    store: ChunkStore | None = None,
) -> int:
    """Embed chunks with BGE-M3 and store them. Return how many were stored.

    Each chunk gets a dense and a sparse vector (``embedding.model``, with
    ``use_fp16`` off on CPU), computed in batches of ``embedding.batch_size``. The
    chunk's metadata is stored alongside, so that search can filter on
    ``access_group`` and answers can cite document and page.

    Indexing a document again replaces its earlier chunks: every chunk already stored
    for a ``doc_id`` that appears in ``chunks`` is deleted first. Pass all of a
    document's chunks in one call.
    """
    if not chunks:
        return 0
    from docqa import resources

    embedder = embedder or resources.get_embedder(config)
    store = store or resources.get_store(config)

    for doc_id in dict.fromkeys(chunk.doc_id for chunk in chunks):
        removed = store.delete_document(doc_id)
        if removed:
            log.info("Replaced %d earlier chunks of document %s", removed, doc_id)

    stored = 0
    for start in range(0, len(chunks), _GROUP_SIZE):
        group = chunks[start : start + _GROUP_SIZE]
        embeddings = embedder.embed([searchable_text(chunk) for chunk in group])
        stored += store.upsert(group, embeddings)
        log.info("Indexed %d of %d chunks", stored, len(chunks))
    return stored


def remove_document(doc_id: str, config: Config, *, store: ChunkStore | None = None) -> int:
    """Delete every chunk of one document from the store. Return how many were removed."""
    from docqa import resources

    store = store or resources.get_store(config)
    return store.delete_document(doc_id)


def count_chunks(config: Config, *, store: ChunkStore | None = None) -> int:
    """Return how many chunks are stored. Zero means nothing has been indexed yet."""
    from docqa import resources

    store = store or resources.get_store(config)
    return store.count()
