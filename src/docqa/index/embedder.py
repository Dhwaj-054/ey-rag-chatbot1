"""Turning text into the two vectors that search uses.

``BGEM3Embedder`` wraps BGE-M3 (through the FlagEmbedding library). For each text it
returns a dense vector, which captures meaning, and a sparse vector, which holds a
weight for each word piece that occurs in the text.

**The model is run from here, not through the library's own call**, when it is on a
CPU. The library's call (``encode``) first runs its longest batch once as a trial, to
find a batch size that fits in a graphics card's memory, throws the result away and
then runs that batch again. On a CPU that trial is pure waste, and ingestion embeds one
document per call: a document of five chunks is two batches, so the trial made that
three. The inputs are prepared the way the library prepares them and the vectors are
taken from the same model call, so they are the same vectors;
``tests/test_direct_models.py`` checks that against the library on a small model, and
a slow test does on the real one. ``docqa.retrieve.reranker`` does the same for the
reranker and gives the measurements.

Anything with the same two methods can stand in for it; the tests use a small fake so
that they need no model.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from docqa.batching import batches_by_length, runs_on_cpu
from docqa.config import EmbeddingConfig
from docqa.quantize import quantize_if_asked

log = logging.getLogger(__name__)


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
        self.direct = runs_on_cpu(self._model)
        """True while the model is run from here and not through the library's call."""

    def embed(self, texts: Sequence[str]) -> list[Embedding]:
        if not texts:
            return []
        if self.direct:
            try:
                batches = self._prepare(texts)
            except Exception as exc:  # noqa: BLE001 - the library's own call still works
                log.warning(
                    "The embedding model could not be run directly and is run through the "
                    "library's own call from now on, which is slower: %s: %s",
                    type(exc).__name__,
                    _first_line(exc),
                )
                self.direct = False
            else:
                try:
                    return self._run(batches, len(texts))
                except Exception as exc:
                    raise EmbeddingError(
                        f"Embedding failed ({type(exc).__name__}: {_first_line(exc)})"
                    ) from exc
        return self._embed_with_library(texts)

    # -- the direct way ---------------------------------------------------------------

    def _prepare(self, texts: Sequence[str]) -> list[tuple[list[int], Any]]:
        """Tokenize as the library does and group the texts into batches of similar length."""
        from FlagEmbedding.utils.tokenizer_compat import pad_with_compat

        tokenizer = self._model.tokenizer
        encoded = tokenizer(list(texts), truncation=True, max_length=self._config.max_length)
        items = [{key: encoded[key][row] for key in encoded.keys()} for row in range(len(texts))]
        batches = []
        for members in batches_by_length([len(item["input_ids"]) for item in items], self._config.batch_size):
            inputs = pad_with_compat(tokenizer, [items[position] for position in members], padding=True, return_tensors="pt")
            batches.append((members, inputs))
        return batches

    def _run(self, batches: list[tuple[list[int], Any]], count: int) -> list[Embedding]:
        import torch

        model = self._model.model
        model.eval()
        ignored = self._tokens_without_weight()
        found: list[Embedding | None] = [None] * count
        with torch.no_grad():
            for members, inputs in batches:
                output = model(
                    inputs,
                    return_dense=True,
                    return_sparse=True,
                    return_colbert_vecs=False,
                    truncate_dim=getattr(self._model, "truncate_dim", None),
                )
                dense = output["dense_vecs"].float().tolist()
                weights = output["sparse_vecs"].squeeze(-1).float().tolist()
                token_ids = inputs["input_ids"].tolist()
                for row, position in enumerate(members):
                    found[position] = Embedding(
                        dense=dense[row], sparse=_token_weights(weights[row], token_ids[row], ignored)
                    )
        if any(item is None for item in found):
            raise EmbeddingError("The embedding model returned fewer vectors than texts.")
        return [item for item in found if item is not None]

    def _tokens_without_weight(self) -> set[int]:
        """The marker tokens (start, end, padding, unknown), which get no sparse weight."""
        tokenizer = self._model.tokenizer
        ignored = set()
        for name in ("cls_token", "eos_token", "pad_token", "unk_token"):
            if name in tokenizer.special_tokens_map:
                ignored.add(tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map[name]))
        return ignored

    # -- the library's way ------------------------------------------------------------

    def _embed_with_library(self, texts: Sequence[str]) -> list[Embedding]:
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


def _token_weights(weights: Sequence[float], token_ids: Sequence[int], ignored: set[int]) -> dict[int, float]:
    """The sparse vector of one text: for each word piece in it, its largest weight."""
    found: dict[int, float] = {}
    for weight, token in zip(weights, token_ids, strict=True):
        if token in ignored or weight <= 0:
            continue
        if weight > found.get(token, 0.0):
            found[token] = float(weight)
    return found


def _first_line(exc: BaseException) -> str:
    lines = str(exc).strip().splitlines()
    return lines[0] if lines else ""
