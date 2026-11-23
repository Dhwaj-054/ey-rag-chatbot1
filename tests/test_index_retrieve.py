"""Indexing and retrieval together.

The embedding model and the reranker are small fakes (tests/fakes.py). The vector store
is the real Qdrant client in local mode, in a temporary folder.
"""

from __future__ import annotations

import pytest

from docqa.config import ConfigError
from docqa.index import count_chunks, index_chunks, remove_document, searchable_text
from docqa.retrieve import rank_candidates, reciprocal_rank_fusion, retrieve
from tests.conftest import TEST_RERANK_THRESHOLD
from tests.fakes import FakeEmbedder, FakeReranker, make_chunk

# -- reciprocal rank fusion -----------------------------------------------------------


def test_fusion_reproduces_the_worked_example_in_design_md():
    """design.md section 8.2: five chunks, k = 60."""
    dense = ["A", "B", "C", "D"]
    sparse = ["C", "A", "E", "B"]

    fused = reciprocal_rank_fusion([dense, sparse], k=60)

    assert [chunk_id for chunk_id, _ in fused] == ["A", "C", "B", "E", "D"]
    scores = dict(fused)
    assert scores["A"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["C"] == pytest.approx(1 / 63 + 1 / 61)
    assert scores["B"] == pytest.approx(1 / 62 + 1 / 64)
    assert scores["E"] == pytest.approx(1 / 63)
    assert scores["D"] == pytest.approx(1 / 64)
    assert [round(score, 4) for _, score in fused] == [0.0325, 0.0323, 0.0318, 0.0159, 0.0156]


def test_a_chunk_found_by_both_searches_beats_one_found_by_a_single_search():
    fused = reciprocal_rank_fusion([["only-dense", "both"], ["both", "only-sparse"]], k=60)
    assert fused[0][0] == "both"


def test_fusion_keeps_first_seen_order_for_equal_scores_and_handles_empty_input():
    assert [cid for cid, _ in reciprocal_rank_fusion([["x"], ["y"]], k=60)] == ["x", "y"]
    assert reciprocal_rank_fusion([], k=60) == []
    assert reciprocal_rank_fusion([[], []], k=60) == []


def test_an_id_listed_twice_in_one_ranking_counts_once():
    fused = dict(reciprocal_rank_fusion([["a", "a", "b"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 63)


# -- indexing -------------------------------------------------------------------------


def test_searchable_text_puts_the_title_and_the_headings_in_front():
    plain = make_chunk(text="The limit is 25 per cent.", doc_title="Exposure Norms")
    in_section = make_chunk(
        text="The limit is 25 per cent.",
        doc_title="Exposure Norms",
        heading_path=("Chapter II", "Exposure limits"),
    )

    assert searchable_text(plain) == "Exposure Norms\nThe limit is 25 per cent."
    assert searchable_text(in_section) == (
        "Exposure Norms\nChapter II > Exposure limits\nThe limit is 25 per cent."
    )


def test_index_chunks_embeds_the_searchable_text_and_stores_every_chunk(tmp_cfg, store):
    embedder = FakeEmbedder()
    chunks = [
        make_chunk("doc1:00000", "Prepayment charges are nil.", heading_path=("Home loans",)),
        make_chunk("doc1:00001", "Branches close at five."),
    ]

    stored = index_chunks(chunks, tmp_cfg, embedder=embedder, store=store)

    assert stored == 2
    assert count_chunks(tmp_cfg, store=store) == 2
    assert embedder.calls == [
        [
            "Lending Policy\nHome loans\nPrepayment charges are nil.",
            "Lending Policy\nBranches close at five.",
        ]
    ]


def test_indexing_a_document_again_replaces_its_earlier_chunks(tmp_cfg, store):
    embedder = FakeEmbedder()
    first = [make_chunk(f"doc1:{n:05d}", f"Old part {n}.") for n in range(3)]
    other = [make_chunk("doc2:00000", "Another document.")]
    index_chunks(first + other, tmp_cfg, embedder=embedder, store=store)

    second = [make_chunk("doc1:00000", "New single part.")]
    index_chunks(second, tmp_cfg, embedder=embedder, store=store)

    assert store.count("doc1") == 1
    assert store.count("doc2") == 1


def test_many_chunks_are_embedded_in_groups(tmp_cfg, store):
    embedder = FakeEmbedder()
    chunks = [make_chunk(f"doc1:{n:05d}", f"Part number {n}.") for n in range(70)]

    assert index_chunks(chunks, tmp_cfg, embedder=embedder, store=store) == 70
    assert [len(call) for call in embedder.calls] == [32, 32, 6]
    assert store.count() == 70


def test_remove_document_and_index_nothing(tmp_cfg, store):
    index_chunks([make_chunk("doc1:00000", "Something.")], tmp_cfg, embedder=FakeEmbedder(), store=store)

    assert index_chunks([], tmp_cfg, embedder=FakeEmbedder(), store=store) == 0
    assert remove_document("doc1", tmp_cfg, store=store) == 1
    assert remove_document("doc1", tmp_cfg, store=store) == 0
    assert count_chunks(tmp_cfg, store=store) == 0


# -- retrieval ------------------------------------------------------------------------

CORPUS = [
    make_chunk("fees:00000", "The processing fee on a home loan is 0.50 percent of the loan amount.", page=12),
    make_chunk("fees:00001", "The processing fee on a personal loan is 2.00 percent.", page=13),
    make_chunk("hours:00000", "Branches open at ten in the morning and close at five.", page=2),
    make_chunk("kyc:00000", "A customer must give proof of identity and proof of address.", page=4),
    make_chunk(
        "risk:00000",
        "The processing fee waiver for staff home loan cases is decided by the risk team.",
        access_group="risk",
        page=7,
    ),
]


@pytest.fixture
def indexed(tmp_cfg, store):
    index_chunks(CORPUS, tmp_cfg, embedder=FakeEmbedder(), store=store)
    return {"embedder": FakeEmbedder(), "store": store, "reranker": FakeReranker()}


def test_retrieve_puts_the_chunk_that_answers_the_question_first(tmp_cfg, indexed):
    result = retrieve("What is the processing fee on a home loan?", tmp_cfg, **indexed)

    assert result.found is True
    assert result.question == "What is the processing fee on a home loan?"
    assert result.chunks[0].chunk.chunk_id == "fees:00000"
    assert result.chunks[0].chunk.page == 12
    assert result.best_score == result.chunks[0].rerank_score
    assert result.best_score >= TEST_RERANK_THRESHOLD
    assert result.elapsed_s > 0
    # Where the time went is recorded, so that the slow step can be seen and tuned.
    timings = result.timings
    assert timings.candidates == len(indexed["reranker"].calls[0][1]) > 0
    assert timings.embed_s >= 0 and timings.search_s >= 0 and timings.rerank_s >= 0
    assert timings.embed_s + timings.search_s + timings.rerank_s <= result.elapsed_s
    rerank_scores = [item.rerank_score for item in result.chunks]
    assert rerank_scores == sorted(rerank_scores, reverse=True)
    assert all(item.fusion_score > 0 for item in result.chunks)


def test_retrieve_refuses_when_even_the_best_chunk_is_weak(tmp_cfg, indexed):
    result = retrieve("Which planets have rings and moons orbiting them?", tmp_cfg, **indexed)

    assert result.found is False
    assert result.best_score is None or result.best_score < TEST_RERANK_THRESHOLD


def test_retrieve_reports_not_found_on_an_empty_index(tmp_cfg, store):
    result = retrieve(
        "Anything at all?", tmp_cfg, embedder=FakeEmbedder(), store=store, reranker=FakeReranker()
    )

    assert result.found is False
    assert result.chunks == ()
    assert result.best_score is None


def test_chunks_outside_the_users_access_groups_never_become_candidates(tmp_cfg, indexed):
    question = "Who decides the processing fee waiver for staff home loan cases?"

    default = retrieve(question, tmp_cfg, **indexed)
    risk_team = retrieve(question, tmp_cfg, ["all", "risk"], **indexed)

    assert "risk:00000" not in [item.chunk.chunk_id for item in default.chunks]
    assert risk_team.chunks[0].chunk.chunk_id == "risk:00000"
    # The reranker was never shown the restricted chunk for the default user.
    default_passages = indexed["reranker"].calls[0][1]
    assert not any("risk team" in passage for passage in default_passages)


def test_retrieve_keeps_top_k_of_the_reranked_candidates(temp_config):
    def two_candidates_one_kept(raw):
        raw["retrieval"]["candidates"] = 2
        raw["retrieval"]["top_k"] = 1

    config = temp_config(two_candidates_one_kept)
    from docqa.index.store import ChunkStore

    store = ChunkStore(config.qdrant)
    try:
        index_chunks(CORPUS, config, embedder=FakeEmbedder(), store=store)
        reranker = FakeReranker()
        result = retrieve(
            "processing fee home loan", config, embedder=FakeEmbedder(), store=store, reranker=reranker
        )
    finally:
        store.close()

    assert len(result.chunks) == 1
    assert len(reranker.calls[0][1]) == 2


def test_retrieve_stops_with_a_configuration_error_until_the_threshold_is_set(temp_config):
    def unset(raw):
        raw["retrieval"]["rerank_score_threshold"] = None

    with pytest.raises(ConfigError, match="rerank_score_threshold"):
        retrieve("Any question", temp_config(unset), embedder=FakeEmbedder(), reranker=FakeReranker())


def test_rank_candidates_works_without_a_threshold_and_rejects_an_empty_question(temp_config):
    def unset(raw):
        raw["retrieval"]["rerank_score_threshold"] = None

    config = temp_config(unset)
    from docqa.index.store import ChunkStore

    store = ChunkStore(config.qdrant)
    try:
        index_chunks(CORPUS, config, embedder=FakeEmbedder(), store=store)
        ranked = rank_candidates(
            "proof of identity", config, embedder=FakeEmbedder(), store=store, reranker=FakeReranker()
        )
        with pytest.raises(ValueError, match="empty"):
            rank_candidates("   ", config, embedder=FakeEmbedder(), store=store, reranker=FakeReranker())
    finally:
        store.close()

    assert ranked[0].chunk.chunk_id == "kyc:00000"
    assert ranked[0].rerank_score == 1.0


# -- progress, and a limit on the time ranking may take ---------------------------------------


class SlowReranker(FakeReranker):
    """Takes "a second" for every passage, on a clock that the test's time module reads."""

    def __init__(self, clock) -> None:
        super().__init__()
        self.clock = clock

    def score(self, question, passages):
        self.clock.now += len(passages)
        return super().score(question, passages)


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def perf_counter(self) -> float:
        return self.now


class BatchingReranker(FakeReranker):
    """Forms its own batches, as the real reranker does: the last passage alone, then the rest."""

    def score_batches(self, question, passages):
        scores = self.score(question, passages)
        last = len(passages) - 1
        yield [(last, scores[last])]
        yield [(position, scores[position]) for position in range(last)]


def test_every_step_of_a_search_is_reported_as_it_happens(tmp_cfg, indexed):
    seen: list[tuple[str, str, dict]] = []

    result = retrieve(
        "What is the processing fee on a home loan?",
        tmp_cfg,
        **indexed,
        progress=lambda step, state, details: seen.append((step, state, details)),
    )

    assert [(step, state) for step, state, _ in seen] == [
        ("question", "start"),
        ("question", "end"),
        ("database", "start"),
        ("database", "end"),
        ("ranking", "start"),
        ("ranking", "progress"),
        ("ranking", "end"),
    ]
    found = result.timings.candidates
    assert seen[3][2]["candidates"] == found and seen[4][2] == {"of": found}
    assert seen[5][2] == {"done": found, "of": found}
    assert seen[6][2]["done"] == found and seen[6][2]["took_s"] == result.timings.rerank_s
    assert result.timings.candidates_found == found  # nothing was left out


def test_candidates_are_scored_a_batch_at_a_time(temp_config, store):
    def batches_of_two(raw):
        raw["reranker"]["batch_size"] = 2

    config = temp_config(batches_of_two)
    index_chunks(CORPUS, config, embedder=FakeEmbedder(), store=store)
    reranker = FakeReranker()
    progress: list[int] = []

    retrieve(
        "What is the processing fee on a home loan?",
        config,
        embedder=FakeEmbedder(),
        store=store,
        reranker=reranker,
        progress=lambda step, state, details: progress.append(details["done"]) if state == "progress" else None,
    )

    assert [len(passages) for _, passages in reranker.calls] == [2, 2]  # four candidates the user may read
    assert progress == [2, 4]


def test_ranking_stops_at_its_time_limit_and_keeps_what_it_has_scored(temp_config, store, monkeypatch):
    def a_short_limit(raw):
        raw["reranker"]["batch_size"] = 1
        raw["retrieval"]["rerank_time_limit_s"] = 2.5

    config = temp_config(a_short_limit)
    index_chunks(CORPUS, config, embedder=FakeEmbedder(), store=store)
    clock = Clock()
    monkeypatch.setattr("docqa.retrieve.time", clock)
    reranker = SlowReranker(clock)
    ends: list[dict] = []

    result = retrieve(
        "What is the processing fee on a home loan?",
        config,
        embedder=FakeEmbedder(),
        store=store,
        reranker=reranker,
        progress=lambda step, state, details: ends.append(details) if (step, state) == ("ranking", "end") else None,
    )

    # A second each: after the third passage 2.5 s have passed, and the fourth is not read.
    assert len(reranker.calls) == 3
    assert (result.timings.candidates, result.timings.candidates_found) == (3, 4)
    assert ends == [{"took_s": 3.0, "done": 3, "of": 4}]
    assert len(result.chunks) == 3 and all(item.rerank_score is not None for item in result.chunks)
    # The passages hybrid search placed highest were the ones that were read.
    assert result.chunks[0].chunk.chunk_id == "fees:00000" and result.found is True


def test_at_least_one_batch_is_scored_however_short_the_limit(temp_config, store, monkeypatch):
    def almost_no_time(raw):
        raw["reranker"]["batch_size"] = 2
        raw["retrieval"]["rerank_time_limit_s"] = 0.001

    config = temp_config(almost_no_time)
    index_chunks(CORPUS, config, embedder=FakeEmbedder(), store=store)
    clock = Clock()
    monkeypatch.setattr("docqa.retrieve.time", clock)

    result = retrieve(
        "What is the processing fee on a home loan?", config, embedder=FakeEmbedder(), store=store, reranker=SlowReranker(clock)
    )

    assert result.timings.candidates == 2 and len(result.chunks) == 2


def test_without_a_limit_every_candidate_is_scored(temp_config, store, monkeypatch):
    def no_limit(raw):
        raw["reranker"]["batch_size"] = 1
        raw["retrieval"]["rerank_time_limit_s"] = None

    config = temp_config(no_limit)
    index_chunks(CORPUS, config, embedder=FakeEmbedder(), store=store)
    clock = Clock()
    monkeypatch.setattr("docqa.retrieve.time", clock)

    result = retrieve(
        "What is the processing fee on a home loan?", config, embedder=FakeEmbedder(), store=store, reranker=SlowReranker(clock)
    )

    assert result.timings.candidates == result.timings.candidates_found == 4


def test_a_reranker_that_forms_its_own_batches_is_asked_for_them(tmp_cfg, indexed):
    """The real reranker groups passages by length. Its scores must land on the right passages."""
    plain = retrieve("What is the processing fee on a home loan?", tmp_cfg, **indexed)
    progress: list[int] = []

    batched = retrieve(
        "What is the processing fee on a home loan?",
        tmp_cfg,
        embedder=indexed["embedder"],
        store=indexed["store"],
        reranker=BatchingReranker(),
        progress=lambda step, state, details: progress.append(details["done"]) if state == "progress" else None,
    )

    assert [(item.chunk.chunk_id, item.rerank_score) for item in batched.chunks] == [
        (item.chunk.chunk_id, item.rerank_score) for item in plain.chunks
    ]
    assert progress == [1, batched.timings.candidates]
