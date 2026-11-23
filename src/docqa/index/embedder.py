"""Turning text into the two vectors that search uses.

``BGEM3Embedder`` wraps BGE-M3 (through the FlagEmbedding library). For each text it
returns a dense vector, which captures meaning, and a sparse vector, which holds a
weight for each word piece that occurs in the text.

Anything with the same two methods can stand in for it; the tests use a small fake so
that they need no model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from docqa.config import EmbeddingConfig
from docqa.quantize import quantize_if_asked


class EmbeddingError(Exception):
    """The embedding model could not be loaded or run."""


@dataclass(frozen=True)
class Embedding:
    """The two vectors of one text."""

    dense: list[float]
    sparse: dict[int, float]
    """Token id -> weight, for the word pieces that occur in the text."""


class Embedder(Protocol):
    """Anything that can embed texts."""

    name: str
    """Model name, for logs."""

    def embed(self, texts: Sequence[str]) -> list[Embedding]:
        """One embedding per text, in the same order."""
        ...


class BGEM3Embedder:
    """BGE-M3 on this machine.

    Three settings differ from the library's defaults, and each one matters:

    - sparse output is switched on (it is off by default);
    - the input limit is ``embedding.max_length`` (the default, 512 tokens, would cut
      chunks of up to 800 tokens without saying so);
    - ``use_fp16`` comes from the configuration and must stay false on a CPU.

    With ``embedding.quantize`` the weights are held as 8-bit numbers, which is faster
    on a CPU and changes the vectors slightly (see ``docqa.quantize``).

    Raises:
        EmbeddingError: The model's files are not on this machine, or loading failed.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        self.name = config.model
        self._config = config
        try:
            from FlagEmbedding import BGEM3FlagModel

            self._model = BGEM3FlagModel(
                config.model,
                use_fp16=config.use_fp16,
                batch_size=config.batch_size,
                query_max_length=config.max_length,
                passage_max_length=config.max_length,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )
        except Exception as exc:  # missing files, out of memory, an import problem
            raise EmbeddingError(
                f"The embedding model {config.model!r} could not be loaded. Is it on this "
                f"machine? ({type(exc).__name__}: {_first_line(exc)})"
            ) from exc
        self.quantized = quantize_if_asked(
            self._model.model, config.quantize, f"The embedding model {config.model}"
        )
        """True if the model runs with 8-bit weights (``embedding.quantize``)."""

    def embed(self, texts: Sequence[str]) -> list[Embedding]:
        if not texts:
            return []
        try:
            output = self._model.encode(
                list(texts),
                batch_size=self._config.batch_size,
                max_length=self._config.max_length,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )
        except Exception as exc:
            raise EmbeddingError(
                f"Embedding failed ({type(exc).__name__}: {_first_line(exc)})"
            ) from exc

        dense_vectors = output["dense_vecs"]
        lexical_weights = output["lexical_weights"]
        if len(dense_vectors) != len(texts) or len(lexical_weights) != len(texts):
            raise EmbeddingError(
                f"The embedding model returned {len(dense_vectors)} dense and "
                f"{len(lexical_weights)} sparse vectors for {len(texts)} texts."
            )
        return [
            Embedding(
                dense=[float(value) for value in dense],
                sparse={int(token): float(weight) for token, weight in weights.items()},
            )
            for dense, weights in zip(dense_vectors, lexical_weights, strict=True)
        ]

    def count_tokens(self, text: str) -> int:
        """Tokens ``text`` takes up in the model's input, special tokens included."""
        return len(self._model.tokenizer.encode(text, add_special_tokens=True))


def _first_line(exc: BaseException) -> str:
    lines = str(exc).strip().splitlines()
    return lines[0] if lines else ""
