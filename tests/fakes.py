"""Small stand-ins for the models, so that tests need no model files and no Ollama.

They are deterministic and crude on purpose. ``FakeEmbedder`` and ``FakeReranker`` only
look at which words two texts share. That is enough to test that the pipeline wires the
real pieces together correctly; it says nothing about search quality.
"""

from __future__ import annotations

import json
import math
import re
import zlib
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path

import httpx

from docqa.config import Config
from docqa.index.embedder import Embedding
from docqa.llm import LLMClient
from docqa.schema import Chunk, ElementType

_WORD = re.compile(r"[a-z0-9]+")


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _slot(word: str, size: int) -> int:
    return zlib.crc32(word.encode("utf-8")) % size


class FakeEmbedder:
    """Dense vector: word counts hashed into a few slots. Sparse vector: word counts."""

    name = "fake-embedder"

    def __init__(self, size: int = 64) -> None:
        self.size = size
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> list[Embedding]:
        self.calls.append(list(texts))
        return [self._one(text) for text in texts]

    def _one(self, text: str) -> Embedding:
        counts = Counter(words(text))
        dense = [0.0] * self.size
        for word, count in counts.items():
            dense[_slot(word, self.size)] += count
        length = math.sqrt(sum(value * value for value in dense)) or 1.0
        return Embedding(
            dense=[value / length for value in dense],
            sparse={_slot(word, 1_000_003): float(count) for word, count in counts.items()},
        )


class FakeReranker:
    """Score: the share of the question's words that the passage contains (0 to 1)."""

    name = "fake-reranker"

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, question: str, passages: Sequence[str]) -> list[float]:
        self.calls.append((question, list(passages)))
        wanted = set(words(question))
        if not wanted:
            return [0.0] * len(passages)
        return [len(wanted & set(words(passage))) / len(wanted) for passage in passages]


def make_chunk(
    chunk_id: str = "doc1:00000",
    text: str = "Some text.",
    *,
    doc_id: str | None = None,
    doc_title: str = "Lending Policy",
    page: int = 1,
    element_type: ElementType = ElementType.TEXT,
    access_group: str = "all",
    version: str | None = None,
    effective_date: date | None = None,
    heading_path: tuple[str, ...] = (),
    html: str | None = None,
    image_path: Path | None = None,
    model_generated: bool = False,
    page_end: int | None = None,
    page_marks: tuple[tuple[int, int], ...] = (),
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id or chunk_id.split(":")[0],
        doc_title=doc_title,
        version=version,
        effective_date=effective_date,
        page=page,
        element_type=element_type,
        access_group=access_group,
        text=text,
        heading_path=heading_path,
        html=html,
        image_path=image_path,
        model_generated=model_generated,
        page_end=page_end,
        page_marks=page_marks,
    )


class ScriptedLLM:
    """A real ``LLMClient`` whose server is faked.

    ``reply`` is the text the model "writes", or a function from the request body to
    that text. ``requests`` records every request body that was sent, so a test can
    check what the model was given, or that it was not called at all.
    """

    def __init__(
        self,
        config: Config,
        reply: str | Callable[[dict], str] = "ok",
        *,
        finish_reason: str = "stop",
        status: int = 200,
    ) -> None:
        self.requests: list[dict] = []
        self._reply = reply
        self._finish_reason = finish_reason
        self._status = status
        self._models = [config.llm.answer_model, config.llm.vision_model]
        self.client = LLMClient(config.llm, transport=httpx.MockTransport(self._handle))

    def _handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            if self._status >= 400:
                raise httpx.ConnectError("connection refused", request=request)
            return httpx.Response(
                200, json={"object": "list", "data": [{"id": name} for name in self._models]}
            )
        body = json.loads(request.content)
        self.requests.append(body)
        if self._status >= 400:
            return httpx.Response(self._status, json={"error": {"message": "scripted failure"}})
        text = self._reply(body) if callable(self._reply) else self._reply
        model = body["model"]
        if body.get("stream"):
            pieces = re.findall(r"\S+\s*", text) or [text]
            events = [
                {"model": model, "choices": [{"delta": {"content": piece}, "finish_reason": None}]}
                for piece in pieces
            ]
            events.append(
                {"model": model, "choices": [{"delta": {}, "finish_reason": self._finish_reason}]}
            )
            events.append(
                {"model": model, "choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 20}}
            )
            lines = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
            return httpx.Response(
                200, content=lines.encode("utf-8"), headers={"content-type": "text/event-stream"}
            )
        return httpx.Response(
            200,
            json={
                "model": model,
                "choices": [
                    {
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": self._finish_reason,
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            },
        )
