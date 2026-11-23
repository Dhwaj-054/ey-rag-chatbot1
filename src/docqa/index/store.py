"""The vector store: one Qdrant collection that holds every chunk.

Each chunk is one point with two named vectors, ``dense`` and ``sparse``, and a payload
that carries all of the chunk's fields. Search can therefore return complete chunks,
and it can be restricted by ``access_group`` inside Qdrant.

Things to know about Qdrant's local mode (``qdrant.path`` set), all seen with
qdrant-client 1.20.0 on Windows:

- Only one process can have the folder open. The chat app and an ingestion run cannot
  both use it. ``StoreError`` says so in plain words.
- ``delete_collection`` leaves the collection's file behind, and a new collection of
  the same name gets the old points back. This class never drops a collection: it
  deletes points by ``doc_id``.
- It keeps every vector in memory and advises against more than 20,000 points.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

from docqa.config import QdrantConfig
from docqa.index.embedder import Embedding
from docqa.schema import Chunk, ElementType

DENSE = "dense"
SPARSE = "sparse"
_UPSERT_BATCH = 64
# Point ids must be UUIDs or integers. This fixed namespace turns a chunk_id into the
# same UUID every time, so indexing a chunk again replaces its point.
_POINT_NAMESPACE = uuid.UUID("7f1d4a3e-5b0c-4f0e-9a51-2d1c6c0a9b11")


class StoreError(Exception):
    """The vector store could not be opened or used."""


class ChunkStore:
    """The Qdrant collection named in ``qdrant.collection``.

    Args:
        config: The ``qdrant`` section of the configuration.

    Raises:
        StoreError: The store could not be opened, for example because another program
            has the local folder open.
    """

    def __init__(self, config: QdrantConfig) -> None:
        from qdrant_client import QdrantClient

        self._collection = config.collection
        self._local = config.path is not None
        try:
            if config.path is not None:
                Path(config.path).mkdir(parents=True, exist_ok=True)
                self._client = QdrantClient(path=str(config.path))
            else:
                self._client = QdrantClient(url=config.url)
        except Exception as exc:
            if "already accessed" in str(exc):
                raise StoreError(
                    f"The index at {config.path} is in use by another program, probably the "
                    f"chat app or an ingestion run. Qdrant's local mode allows one at a "
                    f"time: stop the other one and try again."
                ) from exc
            raise StoreError(
                f"The vector store could not be opened ({type(exc).__name__}: {exc})"
            ) from exc

    def close(self) -> None:
        """Release the store. In local mode this frees the folder for another program."""
        self._client.close()

    # -- writing ----------------------------------------------------------------------

    def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[Embedding]) -> int:
        """Store chunks with their vectors. A chunk already stored is replaced."""
        from qdrant_client import models

        if len(chunks) != len(embeddings):
            raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings.")
        if not chunks:
            return 0
        self._ensure_collection(len(embeddings[0].dense))

        points = [
            models.PointStruct(
                id=point_id(chunk.chunk_id),
                vector={
                    DENSE: embedding.dense,
                    SPARSE: models.SparseVector(
                        indices=list(embedding.sparse.keys()),
                        values=list(embedding.sparse.values()),
                    ),
                },
                payload=chunk_to_payload(chunk),
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        for start in range(0, len(points), _UPSERT_BATCH):
            self._client.upsert(
                collection_name=self._collection, points=points[start : start + _UPSERT_BATCH]
            )
        return len(points)

    def delete_document(self, doc_id: str) -> int:
        """Delete every chunk of one document. Returns how many there were."""
        from qdrant_client import models

        if not self._exists():
            return 0
        before = self.count(doc_id)
        if before:
            self._client.delete(
                collection_name=self._collection,
                points_selector=models.FilterSelector(filter=_doc_filter(doc_id)),
            )
        return before

    # -- reading ----------------------------------------------------------------------

    def count(self, doc_id: str | None = None) -> int:
        """Chunks stored, in all or for one document."""
        if not self._exists():
            return 0
        result = self._client.count(
            collection_name=self._collection,
            count_filter=_doc_filter(doc_id) if doc_id else None,
            exact=True,
        )
        return int(result.count)

    def count_by_type(self) -> dict[str, int]:
        """Chunks stored of each kind: ``{"text": n, "table": n, "figure": n}``."""
        from qdrant_client import models

        counts = {kind.value: 0 for kind in ElementType}
        if not self._exists():
            return counts
        for kind in counts:
            result = self._client.count(
                collection_name=self._collection,
                count_filter=models.Filter(
                    must=[models.FieldCondition(key="element_type", match=models.MatchValue(value=kind))]
                ),
                exact=True,
            )
            counts[kind] = int(result.count)
        return counts

    def search_dense(
        self, vector: Sequence[float], access_groups: Sequence[str], limit: int
    ) -> list[tuple[Chunk, float]]:
        """Chunks closest in meaning to the query vector, best first."""
        return self._search(list(vector), DENSE, access_groups, limit)

    def search_sparse(
        self, weights: dict[int, float], access_groups: Sequence[str], limit: int
    ) -> list[tuple[Chunk, float]]:
        """Chunks that share weighted word pieces with the query, best first."""
        from qdrant_client import models

        if not weights:
            return []
        query = models.SparseVector(indices=list(weights.keys()), values=list(weights.values()))
        return self._search(query, SPARSE, access_groups, limit)

    def _search(
        self, query: Any, using: str, access_groups: Sequence[str], limit: int
    ) -> list[tuple[Chunk, float]]:
        from qdrant_client import models

        if not self._exists():
            return []
        result = self._client.query_points(
            collection_name=self._collection,
            query=query,
            using=using,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="access_group", match=models.MatchAny(any=list(access_groups))
                    )
                ]
            ),
            limit=limit,
            with_payload=True,
        )
        return [(payload_to_chunk(point.payload or {}), float(point.score)) for point in result.points]

    # -- internals --------------------------------------------------------------------

    def _exists(self) -> bool:
        return bool(self._client.collection_exists(self._collection))

    def _ensure_collection(self, dense_size: int) -> None:
        from qdrant_client import models

        if self._exists():
            stored = self._stored_dense_size()
            if stored is not None and stored != dense_size:
                raise StoreError(
                    f"The collection {self._collection!r} holds dense vectors of size "
                    f"{stored}, but the embedding model produces size {dense_size}. The "
                    f"embedding model was changed after documents were indexed. Use a new "
                    f"qdrant.collection name (or a new qdrant.path) and ingest again."
                )
            return
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config={
                DENSE: models.VectorParams(size=dense_size, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={SPARSE: models.SparseVectorParams()},
        )
        if not self._local:
            # Payload indexes make filtering fast on a server. Local mode ignores them.
            for field in ("access_group", "doc_id"):
                self._client.create_payload_index(
                    collection_name=self._collection,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )

    def _stored_dense_size(self) -> int | None:
        vectors = self._client.get_collection(self._collection).config.params.vectors
        if isinstance(vectors, dict) and DENSE in vectors:
            return int(vectors[DENSE].size)
        return None


def point_id(chunk_id: str) -> str:
    """The Qdrant point id for a chunk: always the same for the same ``chunk_id``."""
    return str(uuid.uuid5(_POINT_NAMESPACE, chunk_id))


def chunk_to_payload(chunk: Chunk) -> dict[str, Any]:
    """A chunk as the plain data Qdrant stores next to its vectors."""
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "doc_title": chunk.doc_title,
        "version": chunk.version,
        "effective_date": chunk.effective_date.isoformat() if chunk.effective_date else None,
        "page": chunk.page,
        "page_end": chunk.page_end,
        "page_marks": [list(mark) for mark in chunk.page_marks],
        "element_type": chunk.element_type.value,
        "access_group": chunk.access_group,
        "text": chunk.text,
        "heading_path": list(chunk.heading_path),
        "html": chunk.html,
        "image_path": str(chunk.image_path) if chunk.image_path else None,
        "model_generated": chunk.model_generated,
    }


def payload_to_chunk(payload: dict[str, Any]) -> Chunk:
    """The reverse of ``chunk_to_payload``."""
    effective_date = payload.get("effective_date")
    image_path = payload.get("image_path")
    return Chunk(
        chunk_id=payload["chunk_id"],
        doc_id=payload["doc_id"],
        doc_title=payload["doc_title"],
        version=payload.get("version"),
        effective_date=date.fromisoformat(effective_date) if effective_date else None,
        page=int(payload["page"]),
        element_type=ElementType(payload["element_type"]),
        access_group=payload["access_group"],
        text=payload["text"],
        heading_path=tuple(payload.get("heading_path") or ()),
        html=payload.get("html"),
        image_path=Path(image_path) if image_path else None,
        model_generated=bool(payload.get("model_generated", False)),
        page_end=payload.get("page_end"),
        page_marks=tuple((int(offset), int(page)) for offset, page in payload.get("page_marks") or ()),
    )


def _doc_filter(doc_id: str) -> Any:
    from qdrant_client import models

    return models.Filter(
        must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
    )
