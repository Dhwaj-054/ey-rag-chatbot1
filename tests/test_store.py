"""The vector store, against the real Qdrant client in local mode (a temporary folder)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from docqa.index.store import (
    ChunkStore,
    StoreError,
    chunk_to_payload,
    payload_to_chunk,
    point_id,
)
from docqa.schema import ElementType
from tests.fakes import FakeEmbedder, make_chunk


def put(store: ChunkStore, *chunks) -> None:
    store.upsert(chunks, FakeEmbedder().embed([chunk.text for chunk in chunks]))


def test_an_empty_store_counts_zero_and_finds_nothing(store):
    query = FakeEmbedder().embed(["anything"])[0]

    assert store.count() == 0
    assert store.search_dense(query.dense, ["all"], 5) == []
    assert store.search_sparse(query.sparse, ["all"], 5) == []
    assert store.delete_document("doc1") == 0


def test_dense_and_sparse_search_both_find_the_matching_chunk(store):
    put(
        store,
        make_chunk("doc1:00000", "The processing fee on a home loan is 0.50 percent."),
        make_chunk("doc1:00001", "Branches close at five in the evening."),
        make_chunk("doc1:00002", "Complaints go to the grievance redressal officer."),
    )
    query = FakeEmbedder().embed(["home loan processing fee"])[0]

    dense = store.search_dense(query.dense, ["all"], 3)
    sparse = store.search_sparse(query.sparse, ["all"], 3)

    assert store.count() == 3
    assert dense[0][0].chunk_id == "doc1:00000"
    assert sparse[0][0].chunk_id == "doc1:00000"
    assert dense[0][1] >= dense[-1][1]
    # Sparse search returns only chunks that share a word with the query.
    assert [chunk.chunk_id for chunk, _ in sparse] == ["doc1:00000"]


def test_search_returns_the_whole_chunk(store):
    original = make_chunk(
        "doc1:00007",
        "Table: Charges\nColumns: Loan type | Fee\nHome loan | 0.50%",
        doc_title="Schedule of Charges",
        page=12,
        page_end=13,
        page_marks=((40, 13),),
        element_type=ElementType.TABLE,
        version="v3",
        effective_date=date(2026, 4, 1),
        heading_path=("Chapter II", "Fees"),
        html="<table><tr><td>Home loan</td></tr></table>",
    )
    put(store, original)

    found = store.search_dense(FakeEmbedder().embed(["home loan fee"])[0].dense, ["all"], 1)

    assert found[0][0] == original


def test_only_chunks_in_the_users_access_groups_are_returned(store):
    put(
        store,
        make_chunk("doc1:00000", "Credit policy for everyone.", access_group="all"),
        make_chunk("doc2:00000", "Credit policy for the risk team.", access_group="risk"),
    )
    query = FakeEmbedder().embed(["credit policy"])[0]

    everyone = store.search_dense(query.dense, ["all"], 5)
    risk_team = store.search_dense(query.dense, ["all", "risk"], 5)
    sparse_everyone = store.search_sparse(query.sparse, ["all"], 5)

    assert [chunk.chunk_id for chunk, _ in everyone] == ["doc1:00000"]
    assert {chunk.chunk_id for chunk, _ in risk_team} == {"doc1:00000", "doc2:00000"}
    assert [chunk.chunk_id for chunk, _ in sparse_everyone] == ["doc1:00000"]


def test_storing_a_chunk_again_replaces_it(store):
    put(store, make_chunk("doc1:00000", "Old wording about fees."))
    put(store, make_chunk("doc1:00000", "New wording about fees."))

    found = store.search_dense(FakeEmbedder().embed(["fees"])[0].dense, ["all"], 5)

    assert store.count() == 1
    assert found[0][0].text == "New wording about fees."


def test_deleting_a_document_removes_only_its_chunks(store):
    put(
        store,
        make_chunk("doc1:00000", "First document, part one."),
        make_chunk("doc1:00001", "First document, part two."),
        make_chunk("doc2:00000", "Second document."),
    )

    removed = store.delete_document("doc1")

    assert removed == 2
    assert store.count() == 1
    assert store.count("doc1") == 0
    assert store.count("doc2") == 1


def test_chunks_are_counted_by_kind(store):
    assert store.count_by_type() == {"text": 0, "table": 0, "figure": 0}

    put(
        store,
        make_chunk("doc1:00000", "Some text."),
        make_chunk("doc1:00001", "More text."),
        make_chunk("doc1:00002", "Table: Charges", element_type=ElementType.TABLE),
    )

    assert store.count_by_type() == {"text": 2, "table": 1, "figure": 0}


def test_a_different_vector_size_is_refused_with_an_explanation(store):
    put(store, make_chunk("doc1:00000", "Indexed with the first model."))
    chunk = make_chunk("doc1:00001", "Indexed with another model.")

    with pytest.raises(StoreError, match="embedding model was changed"):
        store.upsert([chunk], FakeEmbedder(size=32).embed([chunk.text]))


def test_number_of_chunks_and_embeddings_must_match(store):
    with pytest.raises(ValueError, match="2 chunks but 1 embeddings"):
        store.upsert(
            [make_chunk("doc1:00000"), make_chunk("doc1:00001")], FakeEmbedder().embed(["one"])
        )


def test_a_second_program_cannot_open_the_same_folder(store, tmp_cfg):
    """Local mode allows one client per folder. The message has to say what to do."""
    with pytest.raises(StoreError, match="in use by another program"):
        ChunkStore(tmp_cfg.qdrant)


def test_the_folder_is_free_again_after_close(tmp_cfg):
    first = ChunkStore(tmp_cfg.qdrant)
    put(first, make_chunk("doc1:00000", "Kept on disk."))
    first.close()

    second = ChunkStore(tmp_cfg.qdrant)
    try:
        assert second.count() == 1
    finally:
        second.close()


def test_payload_round_trip_keeps_every_field():
    chunk = make_chunk(
        "doc9:00003",
        "Figure (bar chart): Net profit",
        element_type=ElementType.FIGURE,
        image_path=Path("data/figures/doc9/p0004-001.png"),
        model_generated=True,
        effective_date=date(2025, 1, 31),
    )

    assert payload_to_chunk(chunk_to_payload(chunk)) == chunk


def test_point_ids_are_stable_and_distinct():
    assert point_id("doc1:00000") == point_id("doc1:00000")
    assert point_id("doc1:00000") != point_id("doc1:00001")
