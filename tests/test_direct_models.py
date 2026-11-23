"""The two search models run from this project's own loop, checked against the library.

A small model of the same kind as each real one is made in a temporary folder, with a
small tokenizer, so no download and no model files are needed. The real FlagEmbedding
classes load it. The same texts then go through this project's loop and through the
library's own call, on the same model object, and must give the same numbers.

What this cannot show is how much time the loop saves with the real models. The
measurement for that is in ``docqa/retrieve/reranker.py``; the slow tests repeat the
comparison of the numbers on the real models.
"""

from __future__ import annotations

import logging

import pytest
import torch

from docqa.batching import batches_by_length, runs_on_cpu
from docqa.config import EmbeddingConfig, RerankerConfig
from docqa.index.embedder import BGEM3Embedder, _token_weights
from docqa.retrieve.reranker import BGEReranker

WORDS = (
    "the processing fee on a home loan is half percent of amount branches close at five in "
    "evening annual report was published june what prepayment charge outstanding principal "
    "levied foreclosure floating rate loans carry no penalty bank must tell borrower before "
    "sanction every year board reviews policy and limit for each sector exposure"
).split()
QUESTION = "what is the processing fee on a home loan"
PASSAGES = [
    "the processing fee on a home loan is half percent of the loan amount",
    "branches close at five in the evening",
    "the annual report was published in june " * 12,  # longer than the input limit: it is cut
    "prepayment charge is levied on the outstanding principal on foreclosure",
    "floating rate loans carry no prepayment penalty",
    "the board reviews the policy and the limit for each sector every year before sanction",
    "home loan",
]


def save_tokenizer(folder) -> int:
    """A word-level tokenizer over WORDS, laid out like XLM-RoBERTa's. Returns its size."""
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    vocabulary = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}
    for word in sorted(set(WORDS)):
        vocabulary[word] = len(vocabulary)
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer.normalizer = normalizers.Lowercase()
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> </s> $B </s>",
        special_tokens=[("<s>", 0), ("</s>", 2)],
    )
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        cls_token="<s>",
        sep_token="</s>",
    ).save_pretrained(folder)
    return len(vocabulary)


def small_config(vocabulary: int):
    from transformers import XLMRobertaConfig

    return XLMRobertaConfig(
        vocab_size=vocabulary,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=130,
        num_labels=1,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
    )


@pytest.fixture(scope="module")
def reranker_folder(tmp_path_factory):
    from transformers import XLMRobertaForSequenceClassification

    folder = tmp_path_factory.mktemp("small-reranker")
    torch.manual_seed(0)
    XLMRobertaForSequenceClassification(small_config(save_tokenizer(folder))).save_pretrained(folder)
    return folder


@pytest.fixture(scope="module")
def embedder_folder(tmp_path_factory):
    from transformers import XLMRobertaModel

    folder = tmp_path_factory.mktemp("small-embedder")
    torch.manual_seed(0)
    XLMRobertaModel(small_config(save_tokenizer(folder))).save_pretrained(folder)
    return folder


def reranker_settings(folder, **changes) -> RerankerConfig:
    settings = {
        "model": str(folder),
        "use_fp16": False,
        "quantize": False,
        "batch_size": 3,
        "max_length": 48,
        "normalize": True,
    }
    return RerankerConfig(**{**settings, **changes})


def embedding_settings(folder, **changes) -> EmbeddingConfig:
    settings = {"model": str(folder), "use_fp16": False, "quantize": False, "batch_size": 3, "max_length": 48}
    return EmbeddingConfig(**{**settings, **changes})


def count_runs(module: torch.nn.Module) -> list[int]:
    """Counts how often ``module`` is run from now on. The count is in the returned list."""
    runs = [0]
    module.register_forward_hook(lambda *_: runs.__setitem__(0, runs[0] + 1))
    return runs


# -- batches ------------------------------------------------------------------------------


def test_batches_hold_inputs_of_similar_length():
    lengths = [10, 90, 12, 85, 11, 88, 50]

    batches = batches_by_length(lengths, 3)

    assert sorted(position for batch in batches for position in batch) == list(range(7))
    assert {frozenset(batch) for batch in batches} == {frozenset({1, 5, 3}), frozenset({6, 2, 4}), frozenset({0})}


def test_the_batch_with_the_first_input_is_run_first():
    """Inputs arrive most promising first. Stopping early must not leave the best one out."""
    lengths = [10, 90, 12, 85, 11, 88, 50]

    batches = batches_by_length(lengths, 3)

    assert 0 in batches[0]
    assert [min(batch) for batch in batches] == sorted(min(batch) for batch in batches)
    assert batches_by_length([], 4) == []
    assert batches_by_length([5], 4) == [[0]]


def test_only_a_model_on_the_processor_is_run_directly():
    class Model:
        def __init__(self, devices):
            self.target_devices = devices

    assert runs_on_cpu(Model(["cpu"])) is True
    assert runs_on_cpu(Model(["cuda:0"])) is False
    assert runs_on_cpu(Model(["cpu", "cpu"])) is False  # the library would start worker processes
    assert runs_on_cpu(object()) is False


# -- the reranker -------------------------------------------------------------------------


@pytest.mark.parametrize("normalize", [True, False])
def test_the_reranker_run_directly_gives_the_library_s_scores(reranker_folder, normalize):
    reranker = BGEReranker(reranker_settings(reranker_folder, normalize=normalize))
    assert reranker.direct is True

    ours = reranker.score(QUESTION, PASSAGES)
    theirs = reranker._score_with_library(QUESTION, PASSAGES)

    assert len(ours) == len(PASSAGES)
    assert ours == pytest.approx(theirs, abs=1e-5)
    assert len(set(round(score, 6) for score in ours)) > 3  # not one number for everything
    if normalize:
        assert all(0.0 <= score <= 1.0 for score in ours)
    assert reranker.score(QUESTION, PASSAGES[:1]) == pytest.approx(theirs[:1], abs=1e-5)
    assert reranker.score(QUESTION, []) == []


def test_the_reranker_runs_each_batch_once_and_the_library_runs_one_twice(reranker_folder):
    """The reason for the direct loop: the library's trial run of its first batch."""
    reranker = BGEReranker(reranker_settings(reranker_folder))
    runs = count_runs(reranker._model.model)

    reranker.score(QUESTION, PASSAGES)
    direct = runs[0]
    reranker._score_with_library(QUESTION, PASSAGES)
    library = runs[0] - direct

    assert direct == 3  # seven passages in batches of three
    assert library == 4


def test_the_reranker_hands_over_its_scores_a_batch_at_a_time(reranker_folder):
    reranker = BGEReranker(reranker_settings(reranker_folder))
    runs = count_runs(reranker._model.model)
    whole = reranker.score(QUESTION, PASSAGES)
    runs[0] = 0

    batches = reranker.score_batches(QUESTION, PASSAGES)
    first = next(batches)

    assert runs[0] == 1  # nothing is computed for a batch before it is asked for
    assert 0 in [position for position, _ in first]
    rest = [pair for batch in batches for pair in batch]
    assert sorted(position for position, _ in [*first, *rest]) == list(range(len(PASSAGES)))
    assert all(score == pytest.approx(whole[position], abs=1e-6) for position, score in [*first, *rest])
    assert list(reranker.score_batches(QUESTION, [])) == []


def test_a_reranker_that_cannot_be_run_directly_uses_the_library(reranker_folder, monkeypatch, caplog):
    reranker = BGEReranker(reranker_settings(reranker_folder))
    expected = reranker._score_with_library(QUESTION, PASSAGES)

    def broken(question, passages):
        raise AttributeError("this version of the library has no such function")

    monkeypatch.setattr(reranker, "_prepare", broken)
    with caplog.at_level(logging.WARNING):
        scores = reranker.score(QUESTION, PASSAGES)

    assert scores == pytest.approx(expected, abs=1e-6)
    assert reranker.direct is False
    assert "run through the library's own call from now on" in caplog.text


def test_the_reranker_with_8_bit_weights_is_still_run_directly(reranker_folder):
    reranker = BGEReranker(reranker_settings(reranker_folder, quantize=True))

    scores = reranker.score(QUESTION, PASSAGES)

    assert reranker.quantized is True and reranker.direct is True
    assert len(scores) == len(PASSAGES) and all(0.0 <= score <= 1.0 for score in scores)
    assert scores == pytest.approx(reranker._score_with_library(QUESTION, PASSAGES), abs=1e-5)


# -- the embedding model ------------------------------------------------------------------


def library_vectors(embedder: BGEM3Embedder, texts: list[str]):
    return embedder._embed_with_library(texts)


def test_the_embedding_model_run_directly_gives_the_library_s_vectors(embedder_folder):
    embedder = BGEM3Embedder(embedding_settings(embedder_folder))
    assert embedder.direct is True
    texts = [QUESTION, *PASSAGES]

    ours = embedder.embed(texts)
    theirs = library_vectors(embedder, texts)

    assert len(ours) == len(texts)
    for mine, other in zip(ours, theirs, strict=True):
        assert mine.dense == pytest.approx(other.dense, abs=1e-5)
        assert mine.sparse.keys() == other.sparse.keys()
        assert all(mine.sparse[token] == pytest.approx(other.sparse[token], abs=1e-5) for token in mine.sparse)
    assert len(ours[0].dense) == 32 and ours[0].dense != ours[1].dense
    assert embedder.embed([QUESTION])[0].dense == pytest.approx(theirs[0].dense, abs=1e-5)
    assert embedder.embed([]) == []


def test_the_embedding_model_runs_each_batch_once_and_the_library_runs_one_twice(embedder_folder):
    embedder = BGEM3Embedder(embedding_settings(embedder_folder))
    runs = count_runs(embedder._model.model)
    texts = PASSAGES[:5]  # a document of five chunks: two batches of three

    embedder.embed(texts)
    direct = runs[0]
    library_vectors(embedder, texts)
    library = runs[0] - direct

    assert direct == 2
    assert library == 3


def test_an_embedding_model_that_cannot_be_run_directly_uses_the_library(embedder_folder, monkeypatch, caplog):
    embedder = BGEM3Embedder(embedding_settings(embedder_folder))
    expected = library_vectors(embedder, PASSAGES)

    def broken(texts):
        raise AttributeError("this version of the library has no such function")

    monkeypatch.setattr(embedder, "_prepare", broken)
    with caplog.at_level(logging.WARNING):
        vectors = embedder.embed(PASSAGES)

    assert [vector.dense for vector in vectors] == [pytest.approx(other.dense, abs=1e-6) for other in expected]
    assert embedder.direct is False
    assert "run through the library's own call from now on" in caplog.text


def test_a_word_piece_keeps_its_largest_weight_and_marker_tokens_get_none():
    weights = [0.9, 0.2, 0.5, 0.0, -0.1, 0.7, 0.3]
    tokens = [0, 17, 17, 42, 43, 2, 1]

    assert _token_weights(weights, tokens, ignored={0, 1, 2}) == {17: 0.5}
    assert _token_weights([0.4, 0.6], [17, 17], ignored=set()) == {17: 0.6}
