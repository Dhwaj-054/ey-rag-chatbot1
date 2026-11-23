"""Stage 3, retrieval: find the chunks that can answer a question.

    question -> embed -> dense search + sparse search -> reciprocal rank fusion
             -> rerank -> keep the best -> refuse if even the best is weak

Each function takes the embedding model, the store and the reranker as optional keyword
arguments. Left out, the shared ones from ``docqa.resources`` are used.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from docqa.config import Config, ConfigError
from docqa.index import searchable_text
from docqa.index.embedder import Embedder
from docqa.index.store import ChunkStore
from docqa.retrieve.reranker import Reranker
from docqa.schema import Chunk, RetrievalResult, RetrievalTimings, ScoredChunk


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int) -> list[tuple[str, float]]:
    """Merge several ranked lists of chunk ids into one.

    Each list adds ``1 / (k + rank)`` to the score of every id it contains, with rank
    counted from 1. Returns ``(chunk_id, score)`` pairs, best first. Ids with the same
    score stay in the order in which they were first seen.

    This is done in Python rather than left to Qdrant, so that it works the same in
    local mode and against a server.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        counted: set[str] = set()
        for rank, chunk_id in enumerate(ranking, start=1):
            if chunk_id in counted:  # listed twice in one ranking: its best position counts
                continue
            counted.add(chunk_id)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def rank_candidates(
    question: str,
    config: Config,
    access_groups: Sequence[str] | None = None,
    *,
    embedder: Embedder | None = None,
    store: ChunkStore | None = None,
    reranker: Reranker | None = None,
) -> list[ScoredChunk]:
    """Hybrid search and reranking, without the refusal decision.

    Returns up to ``retrieval.candidates`` chunks, each with its fusion score and its
    rerank score, best rerank score first. ``retrieve`` builds on this. Evaluation calls
    it directly, because it needs the scores before a threshold has been chosen.
    """
    return _rank(question, config, access_groups, embedder, store, reranker)[0]


def _rank(
    question: str,
    config: Config,
    access_groups: Sequence[str] | None,
    embedder: Embedder | None,
    store: ChunkStore | None,
    reranker: Reranker | None,
) -> tuple[list[ScoredChunk], RetrievalTimings]:
    """As ``rank_candidates``, and also says how long each step took."""
    question = question.strip()
    if not question:
        raise ValueError("The question is empty.")
    from docqa import resources

    embedder = embedder or resources.get_embedder(config)
    store = store or resources.get_store(config)
    groups = list(access_groups) if access_groups else [config.retrieval.default_access_group]
    limit = config.retrieval.candidates

    started = time.perf_counter()
    query = embedder.embed([question])[0]
    embedded = time.perf_counter()

    dense = store.search_dense(query.dense, groups, limit)
    sparse = store.search_sparse(query.sparse, groups, limit)
    chunks: dict[str, Chunk] = {}
    for chunk, _score in (*dense, *sparse):
        chunks.setdefault(chunk.chunk_id, chunk)
    fused = reciprocal_rank_fusion(
        [[chunk.chunk_id for chunk, _ in dense], [chunk.chunk_id for chunk, _ in sparse]],
        config.retrieval.rrf_k,
    )[:limit]
    searched = time.perf_counter()
    if not fused:
        return [], RetrievalTimings(embed_s=embedded - started, search_s=searched - embedded)

    reranker = reranker or resources.get_reranker(config)
    rerank_scores = reranker.score(question, [searchable_text(chunks[cid]) for cid, _ in fused])
    timings = RetrievalTimings(
        embed_s=embedded - started,
        search_s=searched - embedded,
        rerank_s=time.perf_counter() - searched,
        candidates=len(fused),
    )
    scored = [
        ScoredChunk(chunk=chunks[chunk_id], fusion_score=fusion_score, rerank_score=rerank_score)
        for (chunk_id, fusion_score), rerank_score in zip(fused, rerank_scores, strict=True)
    ]
    # sorted() is stable, so chunks with equal rerank scores keep their fusion order.
    return sorted(scored, key=lambda item: item.rerank_score or 0.0, reverse=True), timings


def retrieve(
    question: str,
    config: Config,
    access_groups: Sequence[str] | None = None,
    *,
    embedder: Embedder | None = None,
    store: ChunkStore | None = None,
    reranker: Reranker | None = None,
) -> RetrievalResult:
    """Find the chunks most relevant to a question.

    Runs a dense and a sparse BGE-M3 search, both filtered to ``access_groups``, and
    merges them with reciprocal rank fusion (``retrieval.rrf_k``). The best
    ``retrieval.candidates`` are reranked with ``reranker.model`` and the top
    ``retrieval.top_k`` are returned.

    ``found`` is False when the best rerank score is below
    ``retrieval.rerank_score_threshold``, or when the search returned nothing. The
    caller must then reply with the not-found message and must not call the answer
    model.

    Args:
        question: The user's question.
        config: The loaded configuration.
        access_groups: Groups the user may read. Defaults to
            ``retrieval.default_access_group`` alone.

    Raises:
        ConfigError: ``retrieval.rerank_score_threshold`` is not set.
    """
    threshold = config.retrieval.rerank_score_threshold
    if threshold is None:
        raise ConfigError(
            "retrieval.rerank_score_threshold is not set in config.yaml, so the chatbot "
            "cannot decide when to say that it found nothing. Run "
            "'python -m docqa.evaluate --suggest-threshold' on your test questions and put "
            "the value it proposes in config.yaml."
        )
    started = time.perf_counter()
    ranked, timings = _rank(question, config, access_groups, embedder, store, reranker)
    top = tuple(ranked[: config.retrieval.top_k])
    best_score = top[0].rerank_score if top else None
    return RetrievalResult(
        question=question.strip(),
        chunks=top,
        found=best_score is not None and best_score >= threshold,
        best_score=best_score,
        elapsed_s=time.perf_counter() - started,
        timings=timings,
    )
