"""Tests that need the real models: Ollama, BGE-M3, the reranker and Docling's own models.

    pytest -m slow                      all of them (minutes on a CPU)
    pytest -m slow -k embedding         one group

NONE OF THESE HAS BEEN RUN ON THE MACHINE THEY WERE WRITTEN ON: it had none of the models.
Treat a failure here as "the code or this test needs fixing", and find out which. Run
them one at a time, in the order of this file: each group builds on the one before.

Groups 6 and 7 were added on 2026-11-23 with the speed options and the new chat screen.
Run with ``-s`` to see the times the 8-bit tests print.

What has to be in place first (HANDOVER.md gives the commands):

- Ollama running with ``qwen2.5:7b`` and ``qwen2.5vl:7b``, and its context window
  raised above the default 4,096 tokens;
- ``BAAI/bge-m3`` and ``BAAI/bge-reranker-v2-m3`` in the local Hugging Face cache;
- Docling's models in ``models/docling`` (``parsing.artifacts_path``).

The thresholds in these tests are loose on purpose. They check that each model is wired
up and behaves sensibly, not how good it is. Quality is what ``python -m docqa.evaluate``
measures on real questions.
"""

from __future__ import annotations

import json
import math
import time

import pytest

from docqa.config import Config
from docqa.db import Database
from docqa.generate import answer
from docqa.index.embedder import BGEM3Embedder
from docqa.index.store import ChunkStore
from docqa.ingest import ingest_path
from docqa.ingest.figures import SYSTEM_PROMPT, build_prompt, parse_reply
from docqa.ingest.parser import convert_pdf, needs_ocr
from docqa.llm import LLMClient, user_message
from docqa.retrieve import retrieve
from docqa.retrieve.reranker import BGEReranker
from docqa.schema import ElementType
from docqa.tokens import TokenizerCounter
from tests.pdf_samples import flowchart_png, scanned_pdf, text_pdf

pytestmark = pytest.mark.slow

SENTENCE = "The processing fee on a home loan is 0.50 percent of the loan amount."


# -- helpers --------------------------------------------------------------------------


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True)) / (
        math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    )


@pytest.fixture(scope="module")
def llm(config_for_module: Config):
    with LLMClient(config_for_module.llm) as client:
        yield client


@pytest.fixture(scope="module")
def vision_llm(config_for_module: Config):
    """The server the vision model is on: the same one, unless llm.vision_base_url is set."""
    with LLMClient(config_for_module.llm.for_vision()) as client:
        yield client


@pytest.fixture(scope="module")
def config_for_module() -> Config:
    from docqa.config import load_config
    from tests.conftest import PROJECT_ROOT

    return load_config(PROJECT_ROOT / "config.yaml")


# -- 1. Ollama ------------------------------------------------------------------------


def test_ollama_has_both_models(llm, vision_llm, config_for_module):
    assert llm.has_model(config_for_module.llm.answer_model)
    assert vision_llm.has_model(config_for_module.llm.vision_model)


def test_answer_model_replies_and_streams(llm):
    whole = llm.chat([user_message("Reply with the single word OK.")], max_tokens=10)
    assert "ok" in whole.text.lower()
    assert whole.prompt_tokens and whole.completion_tokens

    stream = llm.chat_stream([user_message("Count from 1 to 5, digits only.")], max_tokens=30)
    pieces = list(stream)
    assert len(pieces) >= 2
    assert "3" in "".join(pieces)
    assert stream.result.finish_reason in ("stop", "length")


def test_ollama_context_window_is_large_enough_for_six_chunks(llm):
    """Ollama cuts a prompt down to its context window without saying so.

    Six chunks of up to 800 tokens plus the instructions need well over the default
    4,096. This sends about 6,000 tokens and checks how many the server counted. If it
    fails, raise the window (OLLAMA_CONTEXT_LENGTH=8192 or more) and restart Ollama.
    """
    filler = " ".join(f"Clause {n} sets a limit of {n} percent." for n in range(650))

    result = llm.chat(
        [user_message(f"{filler}\n\nWhat limit does Clause 7 set? Answer with the number only.")],
        max_tokens=10,
    )

    assert result.prompt_tokens is not None
    assert result.prompt_tokens > 4500, (
        f"The server counted {result.prompt_tokens} prompt tokens, so the prompt was cut. "
        f"Raise Ollama's context window."
    )


def test_vision_model_describes_a_flowchart(vision_llm, tmp_path, config_for_module):
    image = flowchart_png(tmp_path / "flow.png")

    result = vision_llm.describe_image(
        build_prompt(), image, system=SYSTEM_PROMPT, max_tokens=config_for_module.figures.max_tokens
    )
    description = parse_reply(result.text, "0" * 64, config_for_module.llm.vision_model)

    assert description.raw_text is None, f"The reply was not JSON: {result.text[:300]}"
    everything = " ".join([description.figure_type, *description.nodes]).lower()
    assert "flow" in everything or "diagram" in everything
    assert any("credit" in node.lower() for node in description.nodes)
    assert description.edges, "No arrows were reported."


# -- 2. Embedding model ---------------------------------------------------------------


@pytest.fixture(scope="module")
def embedder(config_for_module):
    return BGEM3Embedder(config_for_module.embedding)


def test_embedding_gives_a_dense_and_a_sparse_vector(embedder):
    first, second = embedder.embed([SENTENCE, "Branches close at five in the evening."])

    assert len(first.dense) == len(second.dense) == 1024
    assert first.sparse and all(weight > 0 for weight in first.sparse.values())
    assert first.dense != second.dense


def test_embedding_places_related_texts_closer_than_unrelated_ones(embedder):
    query, related, unrelated = embedder.embed(
        [
            "What is the penalty for paying a loan off early?",
            "Prepayment charges: 3 percent of the outstanding principal is levied on foreclosure.",
            "Branches are open from ten in the morning to five in the evening.",
        ]
    )

    assert cosine(query.dense, related.dense) > cosine(query.dense, unrelated.dense)


def test_embedding_reads_past_the_first_512_tokens(embedder, config_for_module):
    """The library's own limit is 512 tokens. The end of a long chunk must still count."""
    padding = " ".join(f"Paragraph {n} is about staff canteen timings." for n in range(90))
    with_ending = f"{padding} The prepayment charge on a personal loan is 3 percent."
    counter = TokenizerCounter(config_for_module.embedding.model)
    assert 512 < counter.count(with_ending) <= config_for_module.embedding.max_length

    plain, ending = embedder.embed([padding, with_ending])

    assert plain.dense != ending.dense
    assert len(ending.sparse) > len(plain.sparse)


# -- 3. Reranker ----------------------------------------------------------------------


def test_reranker_scores_the_answering_passage_highest(config_for_module):
    reranker = BGEReranker(config_for_module.reranker)

    scores = reranker.score(
        "What is the processing fee on a home loan?",
        [SENTENCE, "Branches close at five in the evening.", "The annual report was published in June."],
    )

    assert len(scores) == 3
    assert scores[0] == max(scores)
    if config_for_module.reranker.normalize:
        assert all(0.0 <= score <= 1.0 for score in scores)
        assert scores[0] > 0.5 > scores[1]
    assert reranker.score("One passage only?", ["Yes."])[0] == pytest.approx(
        reranker.score("One passage only?", ["Yes.", "No."])[0], abs=1e-3
    )


# -- 4. Docling -----------------------------------------------------------------------


def test_docling_parses_a_pdf_with_a_text_layer(tmp_path, config_for_module):
    pdf = text_pdf(tmp_path / "policy.pdf", ["Schedule of charges", SENTENCE, "Prepayment is free."])

    document = convert_pdf(pdf, config_for_module)

    texts = " ".join(getattr(item, "text", "") for item, _ in document.iterate_items())
    assert "processing fee on a home loan" in texts
    assert len(document.pages) == 1
    assert all(item.prov and item.prov[0].page_no == 1 for item, _ in document.iterate_items())


def test_docling_reads_a_scanned_page_with_ocr(tmp_path, config_for_module):
    pdf = scanned_pdf(tmp_path / "scan.pdf", ["CIRCULAR ON PREPAYMENT", "No charge applies to floating rate loans."])

    document = convert_pdf(pdf, config_for_module)

    texts = " ".join(getattr(item, "text", "") for item, _ in document.iterate_items()).lower()
    assert "prepayment" in texts or "floating" in texts, f"OCR found: {texts[:300]!r}"


# -- 5. Everything together -----------------------------------------------------------


def test_a_document_goes_from_pdf_to_a_cited_answer(tmp_path, temp_config, llm):
    """Ingest one small PDF with the real models, ask about it, and get its page cited."""

    def change(raw):
        from tests.conftest import PROJECT_ROOT

        # The temporary project uses the real Docling models of the real project.
        if raw["parsing"]["artifacts_path"]:
            raw["parsing"]["artifacts_path"] = str(PROJECT_ROOT / raw["parsing"]["artifacts_path"])
        raw["retrieval"]["rerank_score_threshold"] = 0.2
        raw["figures"]["describe"] = False

    config = temp_config(change)
    config.paths.docs_dir.mkdir(parents=True)
    text_pdf(
        config.paths.docs_dir / "Schedule of Charges.pdf",
        ["Schedule of charges for retail loans", SENTENCE, "The fee is collected when the loan is sanctioned."],
    )

    store = ChunkStore(config.qdrant)
    try:
        report = ingest_path(config.paths.docs_dir, config, store=store, db=Database(config.paths.sqlite_path))
        assert report.failures == ()
        assert report.chunks_indexed >= 1

        asked = retrieve("What is the processing fee on a home loan?", config, store=store)
        refused = retrieve("Which planets in the solar system have rings?", config, store=store)
        result = answer("What is the processing fee on a home loan?", asked, llm, config)
    finally:
        store.close()

    assert asked.found, f"best score {asked.best_score}"
    assert asked.chunks[0].chunk.element_type is ElementType.TEXT
    assert asked.chunks[0].chunk.page == 1
    assert not refused.found, f"best score {refused.best_score}"
    assert "0.50" in result.text
    assert [(c.doc_title, c.page) for c in result.citations] == [("Schedule of Charges", 1)], result.text
    assert result.unmatched_citations == ()


# -- 6. The speed options ---------------------------------------------------------------


def test_a_pdf_with_a_text_layer_is_parsed_without_ocr(tmp_path, config_for_module):
    """parsing.ocr: auto. The check finds the text layer, and Docling runs without its OCR model."""
    plain = text_pdf(tmp_path / "policy.pdf", ["Schedule of charges", SENTENCE])
    scan = scanned_pdf(tmp_path / "scan.pdf", ["CIRCULAR ON PREPAYMENT"])
    assert needs_ocr(plain, config_for_module) is (config_for_module.parsing.ocr is True)
    assert needs_ocr(scan, config_for_module) is (config_for_module.parsing.ocr is not False)

    document = convert_pdf(plain, config_for_module, ocr=False)

    texts = " ".join(getattr(item, "text", "") for item, _ in document.iterate_items())
    assert "processing fee on a home loan" in texts


def timed(label: str, work):
    started = time.perf_counter()
    result = work()
    print(f"\n    {label}: {time.perf_counter() - started:.1f} s")
    return result


def test_the_8_bit_embedding_model_agrees_with_the_full_one(embedder, config_for_module):
    """embedding.quantize. Faster on a CPU; the vectors change a little, the ordering must not."""
    texts = [
        "What is the penalty for paying a loan off early?",
        "Prepayment charges: 3 percent of the outstanding principal is levied on foreclosure.",
        "Branches are open from ten in the morning to five in the evening.",
    ] * 4
    small = BGEM3Embedder(config_for_module.embedding.model_copy(update={"quantize": True}))
    assert small.quantized, "8-bit weights could not be switched on. The log says why."

    full_vectors = timed("full embedding model, 12 texts", lambda: embedder.embed(texts))
    small_vectors = timed("8-bit embedding model, 12 texts", lambda: small.embed(texts))

    query, related, unrelated = small_vectors[:3]
    assert cosine(query.dense, related.dense) > cosine(query.dense, unrelated.dense)
    for full, eight_bit in zip(full_vectors[:3], small_vectors[:3], strict=True):
        assert cosine(full.dense, eight_bit.dense) > 0.9
        assert eight_bit.sparse


def test_the_8_bit_reranker_still_ranks_the_answering_passage_first(config_for_module):
    """reranker.quantize. Its scores differ, so the threshold has to be chosen again with it."""
    passages = [SENTENCE, "Branches close at five in the evening.", "The annual report was published in June."] * 4
    question = "What is the processing fee on a home loan?"
    full = BGEReranker(config_for_module.reranker.model_copy(update={"quantize": False}))
    small = BGEReranker(config_for_module.reranker.model_copy(update={"quantize": True}))
    assert small.quantized, "8-bit weights could not be switched on. The log says why."

    full_scores = timed("full reranker, 12 passages", lambda: full.score(question, passages))
    small_scores = timed("8-bit reranker, 12 passages", lambda: small.score(question, passages))

    assert small_scores[0] == max(small_scores)
    assert full_scores[0] == max(full_scores)
    if config_for_module.reranker.normalize:
        assert small_scores[0] > 0.5 > small_scores[1]


# -- 7. The chat screen -----------------------------------------------------------------


def test_the_chat_screen_answers_a_question_with_the_real_models(tmp_path, temp_config):
    """Start the web server over one small ingested PDF and ask through its own address."""
    from starlette.testclient import TestClient

    from docqa import resources
    from docqa.web.server import create_app

    def change(raw):
        from tests.conftest import PROJECT_ROOT

        if raw["parsing"]["artifacts_path"]:
            raw["parsing"]["artifacts_path"] = str(PROJECT_ROOT / raw["parsing"]["artifacts_path"])
        raw["retrieval"]["rerank_score_threshold"] = 0.2
        raw["figures"]["describe"] = False

    config = temp_config(change)
    config.paths.docs_dir.mkdir(parents=True)
    text_pdf(
        config.paths.docs_dir / "Schedule of Charges.pdf",
        ["Schedule of charges for retail loans", SENTENCE, "The fee is collected when the loan is sanctioned."],
    )
    report = ingest_path(config.paths.docs_dir, config)
    assert report.failures == ()
    resources.release("store")  # the server opens the index itself

    with TestClient(create_app(config, warm_up=False)) as client:
        documents = client.get("/api/documents").json()["documents"]
        response = client.post("/api/ask", json={"question": "What is the processing fee on a home loan?"})
        stats = client.get("/api/stats").json()

    events = []
    for block in response.text.split("\n\n"):
        name = next((line[6:].strip() for line in block.splitlines() if line.startswith("event:")), None)
        data = next((line[5:].strip() for line in block.splitlines() if line.startswith("data:")), None)
        if name:
            events.append((name, json.loads(data) if data else None))
    names = [name for name, _ in events]
    final = next(data for name, data in events if name == "answer")

    assert [doc["title"] for doc in documents] == ["Schedule of Charges"]
    assert "error" not in names, events
    assert names.index("retrieval") < names.index("token") < names.index("answer")
    assert "0.50" in final["text"]
    assert [(c["doc_title"], c["page"]) for c in final["citations"]] == [("Schedule of Charges", 1)], final["text"]
    assert final["timings"]["first_token_s"] > 0 and final["timings"]["retrieval"]["rerank_s"] > 0
    assert final["timings"]["retrieval"]["search_s"] < 1.0  # the database is not the slow part
    assert stats["queries"]["answered"] == 1
