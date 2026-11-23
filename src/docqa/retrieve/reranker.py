"""Scoring how well a passage answers a question.

``BGEReranker`` wraps ``bge-reranker-v2-m3`` (through the FlagEmbedding library). It
reads the question and one passage together and returns one number per passage. That
is slower and more accurate than comparing embeddings, so it runs only on the
candidates that hybrid search returned.

Anything with the same method can stand in for it; the tests use a small fake.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from docqa.config import RerankerConfig
from docqa.quantize import quantize_if_asked


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

    def score(self, question: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
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


def _first_line(exc: BaseException) -> str:
    lines = str(exc).strip().splitlines()
    return lines[0] if lines else ""
