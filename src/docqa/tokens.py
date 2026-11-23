"""Counting tokens, for chunk sizes.

Chunk sizes in ``config.yaml`` are in tokens of the embedding model, because that
model's input limit is the one that matters for search. ``TokenizerCounter`` counts
with the embedding model's own tokenizer, which needs the model's files on disk (not
its weights in memory: loading a tokenizer takes a second or two).

``WordCounter`` counts words. Tests use it, and it is there for anyone who wants to
try the chunker before the model has been downloaded. It is not used by default,
because a silent fallback would give different chunks on different machines.
"""

from __future__ import annotations

from typing import Protocol


class TokenCounterError(Exception):
    """The tokenizer could not be loaded."""


class TokenCounter(Protocol):
    """Anything that can say how many tokens a text has."""

    def count(self, text: str) -> int: ...


class WordCounter:
    """Counts whitespace-separated words, scaled by a fixed number of tokens per word."""

    def __init__(self, tokens_per_word: float = 1.0) -> None:
        if tokens_per_word <= 0:
            raise ValueError("tokens_per_word must be positive.")
        self._tokens_per_word = tokens_per_word

    def count(self, text: str) -> int:
        words = len(text.split())
        return int(words * self._tokens_per_word + 0.999999) if words else 0


class TokenizerCounter:
    """Counts with the tokenizer of a Hugging Face model (the embedding model).

    Args:
        model: Model name or local folder, as in ``embedding.model``.

    Raises:
        TokenCounterError: The tokenizer's files are not on this machine.
    """

    def __init__(self, model: str) -> None:
        try:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(model)
        except Exception as exc:  # missing files, a broken cache, an import problem
            raise TokenCounterError(
                f"The tokenizer of {model!r} could not be loaded, so chunk sizes cannot be "
                f"measured. Is the embedding model on this machine? "
                f"({type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''})"
            ) from exc
        # Counting never feeds the model, so the model's own length limit does not apply
        # and its "sequence is longer than the maximum" warning would only be noise.
        self._tokenizer.model_max_length = 10**9

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._tokenizer.encode(text, add_special_tokens=False))
