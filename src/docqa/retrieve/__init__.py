"""Stage 3, retrieval: find the chunks that can answer a question.

    question -> embed -> dense search + sparse search -> reciprocal rank fusion
             -> rerank -> keep the best -> refuse if even the best is weak

Each function takes the embedding model, the store and the reranker as optional keyword
arguments. Left out, the shared ones from ``docqa.resources`` are used.

Reranking is the slow part of a search. Every candidate is scored, a batch at a time,
and a ``progress`` function, if one is given, is told as each step starts and ends and
after every batch, so that a screen can show where a question is.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from typing import Any

from docqa.config import Config, ConfigError
from docqa.index import searchable_text
from docqa.index.embedder import Embedder
from docqa.index.store import ChunkStore
from docqa.retrieve.reranker import Reranker, RerankError
from docqa.schema import Chunk, RetrievalResult, RetrievalTimings, ScoredChunk

Progress = Callable[[str, str, dict[str, Any]], None]
"""``progress(step, state, details)``. Steps: ``question`` (it is turned into vectors),
``database`` (the two searches) and ``ranking``. States: ``start``, ``progress`` (ranking
only, after each batch) and ``end``."""


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
    progress: Progress | None = None,
) -> tuple[list[ScoredChunk], RetrievalTimings]:
    """As ``rank_candidates``, and also says how long each step took."""
    question = question.strip()
    if not question:
        raise ValueError("The question is empty.")
    from docqa import resources

    tell = progress or (lambda step, state, details: None)
    embedder = embedder or resources.get_embedder(config)
    store = store or resources.get_store(config)
    groups = list(access_groups) if access_groups else [config.retrieval.default_access_group]
    limit = config.retrieval.candidates

    started = time.perf_counter()
    tell("question", "start", {})
    query = embedder.embed([question])[0]
    embedded = time.perf_counter()
    tell("question", "end", {"took_s": embedded - started})

    tell("database", "start", {})
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
    tell("database", "end", {"took_s": searched - embedded, "candidates": len(fused)})
    if not fused:
        return [], RetrievalTimings(embed_s=embedded - started, search_s=searched - embedded)

    reranker = reranker or resources.get_reranker(config)
    tell("ranking", "start", {"of": len(fused)})
    rerank_scores = _score_in_batches(
        reranker,
        question,
        [searchable_text(chunks[chunk_id]) for chunk_id, _ in fused],
        config.reranker.batch_size,
        tell,
    )
    ranked_s = time.perf_counter() - searched
    tell("ranking", "end", {"took_s": ranked_s, "done": len(fused), "of": len(fused)})
    timings = RetrievalTimings(
        embed_s=embedded - started,
        search_s=searched - embedded,
        rerank_s=ranked_s,
        candidates=len(fused),
    )
    scored = [
        ScoredChunk(chunk=chunks[chunk_id], fusion_score=fusion_score, rerank_score=rerank_score)
        for (chunk_id, fusion_score), rerank_score in zip(fused, rerank_scores, strict=True)
    ]
    # sorted() is stable, so chunks with equal rerank scores keep their fusion order.
    return sorted(scored, key=lambda item: item.rerank_score or 0.0, reverse=True), timings


def _score_in_batches(
    reranker: Reranker,
    question: str,
    passages: Sequence[str],
    batch_size: int,
    tell: Progress,
) -> list[float]:
    """Score every passage, a batch at a time, saying after each batch how far it is.

    Returns one score per passage, in the order of ``passages``.
    """
    scores: dict[int, float] = {}
    for batch in _batches(reranker, question, passages, batch_size):
        scores.update(batch)
        tell("ranking", "progress", {"done": len(scores), "of": len(passages)})
    if len(scores) != len(passages):
        raise RerankError(f"The reranker returned {len(scores)} scores for {len(passages)} passages.")
    return [scores[position] for position in range(len(passages))]


def _batches(
    reranker: Reranker, question: str, passages: Sequence[str], batch_size: int
) -> Iterator[list[tuple[int, float]]]:
    """The reranker's scores, ``(position, score)`` pairs a batch at a time.

    The real reranker forms the batches itself, by length. Any other one (the tests'
    stand-in) is given the passages in slices.
    """
    in_batches = getattr(reranker, "score_batches", None)
    if in_batches is not None:
        yield from in_batches(question, passages)
        return
    for start in range(0, len(passages), batch_size):
        part = reranker.score(question, passages[start : start + batch_size])
        yield [(start + offset, score) for offset, score in enumerate(part)]


def retrieve(
    question: str,
    config: Config,
    access_groups: Sequence[str] | None = None,
    *,
    embedder: Embedder | None = None,
    store: ChunkStore | None = None,
    reranker: Reranker | None = None,
    progress: Progress | None = None,
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
        progress: Told as each step starts and ends (see ``Progress``).

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
    ranked, timings = _rank(question, config, access_groups, embedder, store, reranker, progress)
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
