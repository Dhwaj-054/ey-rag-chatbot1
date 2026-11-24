"""Scoring how well a passage answers a question.

``BGEReranker`` wraps ``bge-reranker-v2-m3`` (through the FlagEmbedding library). It
reads the question and one passage together and returns one number per passage. That
is slower and more accurate than comparing embeddings, so it runs only on the
candidates that hybrid search returned.

**The model is run from here, not through the library's own scoring call**, when it is
on a CPU. The library's call (``compute_score``) first runs its longest batch once as a
trial, to find a batch size that fits in a graphics card's memory, throws the result
away and then runs that batch again. On a CPU there is no such memory limit to find,
and that trial was a quarter of the ranking time: measured on a laptop with a stand-in
model of the same size, 15 passages took 42 s with the library's call and 33 s without
the trial (22 s and 14 s with 8-bit weights). The inputs are prepared with the
library's own functions, so the scores are the same; ``tests/test_direct_models.py``
checks that against the library on a small model, and a slow test does on the real one.

Anything with a ``score`` method can stand in for the reranker; the tests use a small
fake.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator, Sequence
from typing import Any, Protocol

from docqa.batching import batches_by_length, runs_on_cpu
from docqa.config import RerankerConfig
from docqa.quantize import quantize_if_asked

log = logging.getLogger(__name__)


class RerankError(Exception):
    """The reranker could not be loaded or run."""


class Reranker(Protocol):
    """Anything that can score passages against a question."""

    name: str
    """Model name, for logs."""

    def score(self, question: str, passages: Sequence[str]) -> list[float]:
        """One score per passage, in the same order. Higher is more relevant."""
        ...


class BGEReranker:
    """bge-reranker-v2-m3 on this machine.

    Two settings differ from what the library does when left alone:

    - the input limit is ``reranker.max_length`` (the default, 512 tokens for question
      and passage together, would trim chunks of up to 800 tokens);
    - ``reranker.normalize`` decides the scale of the scores. With it on, they lie
      between 0 and 1. ``retrieval.rerank_score_threshold`` is on whichever scale this
      setting produces, so changing one means setting the other again.

    Raises:
        RerankError: The model's files are not on this machine, or loading failed.
    """

    def __init__(self, config: RerankerConfig) -> None:
        self.name = config.model
        self._config = config
        try:
            from FlagEmbedding import FlagReranker

            self._model = FlagReranker(
                config.model,
                use_fp16=config.use_fp16,
                batch_size=config.batch_size,
                max_length=config.max_length,
                normalize=config.normalize,
            )
        except Exception as exc:  # missing files, out of memory, an import problem
            raise RerankError(
                f"The reranker {config.model!r} could not be loaded. Is it on this machine? "
                f"({type(exc).__name__}: {_first_line(exc)})"
            ) from exc
        self.quantized = quantize_if_asked(
            self._model.model, config.quantize, f"The reranker {config.model}"
        )
        """True if the model runs with 8-bit weights (``reranker.quantize``)."""
        self.direct = runs_on_cpu(self._model)
        """True while the model is run from here and not through the library's call."""

    def score(self, question: str, passages: Sequence[str]) -> list[float]:
        scores = [0.0] * len(passages)
        for batch in self.score_batches(question, passages):
            for position, value in batch:
                scores[position] = value
        return scores

    def score_batches(self, question: str, passages: Sequence[str]) -> Iterator[list[tuple[int, float]]]:
        """Score the passages a batch at a time: ``(position, score)`` pairs per batch.

        A caller that shows progress takes the batches one by one; nothing is computed
        for a batch until it is asked for. A batch holds passages of similar length, so
        that little padding is computed. The batch that holds the first passage comes
        first, then the one that holds the earliest passage not yet scored, and so on.
        """
        if not passages:
            return
        if self.direct:
            try:
                batches = self._prepare(question, passages)
            except Exception as exc:  # noqa: BLE001 - the library's own call still works
                log.warning(
                    "The reranker could not be run directly and is run through the library's "
                    "own call from now on, which is slower: %s: %s",
                    type(exc).__name__,
                    _first_line(exc),
                )
                self.direct = False
            else:
                try:
                    for members, inputs in batches:
                        yield list(zip(members, self._run(inputs), strict=True))
                except Exception as exc:
                    raise RerankError(
                        f"Reranking failed ({type(exc).__name__}: {_first_line(exc)})"
                    ) from exc
                return
        yield list(enumerate(self._score_with_library(question, passages)))

    # -- the direct way ---------------------------------------------------------------

    def _prepare(self, question: str, passages: Sequence[str]) -> list[tuple[list[int], Any]]:
        """Tokenize as the library does, and group the passages into batches.

        Returns ``(positions, padded inputs)`` per batch, in the order to run them.
        """
        from FlagEmbedding.utils.tokenizer_compat import pad_with_compat, prepare_for_model_compat

        tokenizer = self._model.tokenizer
        max_length = self._config.max_length
        query_max_length = getattr(self._model, "query_max_length", None) or max_length * 3 // 4
        question_ids = tokenizer(
            [question], return_tensors=None, add_special_tokens=False, max_length=query_max_length, truncation=True
        )["input_ids"][0]
        passage_ids = tokenizer(
            list(passages), return_tensors=None, add_special_tokens=False, max_length=max_length, truncation=True
        )["input_ids"]
        items = [
            prepare_for_model_compat(
                tokenizer, question_ids, ids, truncation="only_second", max_length=max_length, padding=False
            )
            for ids in passage_ids
        ]
        batches = []
        for members in batches_by_length([len(item["input_ids"]) for item in items], self._config.batch_size):
            inputs = pad_with_compat(tokenizer, [items[position] for position in members], padding=True, return_tensors="pt")
            batches.append((members, inputs))
        return batches

    def _run(self, inputs: Any) -> list[float]:
        import torch

        model = self._model.model
        model.eval()
        with torch.no_grad():
            logits = model(**inputs, return_dict=True).logits.view(-1).float().tolist()
        return [_sigmoid(value) for value in logits] if self._config.normalize else logits

    # -- the library's way ------------------------------------------------------------

    def _score_with_library(self, question: str, passages: Sequence[str]) -> list[float]:
        try:
            scores = self._model.compute_score(
                [(question, passage) for passage in passages],
                batch_size=self._config.batch_size,
                max_length=self._config.max_length,
                normalize=self._config.normalize,
            )
        except Exception as exc:
            raise RerankError(
                f"Reranking failed ({type(exc).__name__}: {_first_line(exc)})"
            ) from exc
        if isinstance(scores, (int, float)):  # the library returns a bare number for one pair
            scores = [scores]
        scores = [float(score) for score in scores]
        if len(scores) != len(passages):
            raise RerankError(
                f"The reranker returned {len(scores)} scores for {len(passages)} passages."
            )
        return scores


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    grown = math.exp(value)
    return grown / (1.0 + grown)


def _first_line(exc: BaseException) -> str:
    lines = str(exc).strip().splitlines()
    return lines[0] if lines else ""
