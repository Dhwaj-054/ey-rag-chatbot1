"""The one module through which this project talks to a language model.

It speaks the OpenAI-compatible HTTP API (``/v1/chat/completions`` and ``/v1/models``)
that Ollama serves on this machine and that vLLM serves on a GPU server, so moving
from one to the other is a change to ``config.yaml`` only.

Plain ``httpx`` is used rather than a vendor SDK: there is no built-in cloud address to
fall back to, and an address that is not this machine is refused unless
``llm.allow_remote`` is set in the configuration.

The vision model may be served somewhere else than the answer model (vLLM serves one
model per server). ``LLMClient(config.llm)`` talks to the answer model's server and
``LLMClient(config.llm.for_vision())`` to the vision model's.

Typical use::

    from docqa.config import load_config
    from docqa.llm import LLMClient, system_message, user_message

    config = load_config()
    with LLMClient(config.llm) as llm:
        reply = llm.chat([system_message("Answer briefly."), user_message("Hello")])
        print(reply.text)

        stream = llm.chat_stream([user_message("Hello")])
        for piece in stream:
            print(piece, end="")
        print(stream.result.latency_s)
"""

from __future__ import annotations

import base64
import ipaddress
import json
import mimetypes
import os
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self
from urllib.parse import urlsplit

import httpx

from docqa.config import LLMConfig

Message = dict[str, Any]
"""One chat message in OpenAI format: ``{"role": ..., "content": ...}``."""

# Image formats that Ollama's OpenAI-compatible endpoint accepts, as base64 data URLs.
# Source: decodeImageURL in ollama/openai/openai.go, read on 2026-11-18. It does not
# accept links to images.
SUPPORTED_IMAGE_TYPES = frozenset({"image/png", "image/jpeg", "image/webp"})

_RESERVED_PAYLOAD_KEYS = frozenset({"model", "messages", "stream"})
_DONE = object()


class LLMError(Exception):
    """Base class for every failure raised by this module."""


class LLMConfigError(LLMError):
    """The ``llm`` settings cannot be used, for example a remote address that is not allowed."""


class LLMConnectionError(LLMError):
    """The server could not be reached, or the connection broke."""


class LLMTimeoutError(LLMError):
    """The server was reached but did not answer in time."""


class LLMResponseError(LLMError):
    """The server answered with an error, or with something that could not be understood."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class ChatResult:
    """A complete reply from the model."""

    text: str
    model: str
    """Model name as reported by the server."""
    finish_reason: str | None
    """``"stop"`` for a normal end. ``"length"`` means the reply hit ``max_tokens`` and is cut off."""
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_s: float
    """Seconds from sending the request to receiving the end of the reply."""


def system_message(text: str) -> Message:
    """Instructions for the model."""
    return {"role": "system", "content": text}


def user_message(text: str) -> Message:
    """A plain-text message from the user."""
    return {"role": "user", "content": text}


def image_message(text: str, image: bytes, mime_type: str = "image/png") -> Message:
    """A user message made of text plus one image, for the vision model.

    The image travels inside the request as base64. PNG, JPEG and WebP are accepted.
    """
    if mime_type not in SUPPORTED_IMAGE_TYPES:
        allowed = ", ".join(sorted(SUPPORTED_IMAGE_TYPES))
        raise ValueError(f"Unsupported image type {mime_type!r}. Use one of: {allowed}.")
    if not image:
        raise ValueError("The image is empty.")
    encoded = base64.b64encode(image).decode("ascii")
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}},
        ],
    }


def is_local_url(url: str) -> bool:
    """True if ``url`` points at this machine (``localhost`` or a loopback address)."""
    host = urlsplit(url).hostname
    if not host:
        return False
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def model_is_listed(name: str, available: Iterable[str]) -> bool:
    """True if ``name`` is among the model names a server reported.

    Ollama treats a name without a tag as ``name:latest``, so ``llama3`` and
    ``llama3:latest`` count as the same model.
    """
    wanted = _with_default_tag(name)
    return any(_with_default_tag(model) == wanted for model in available)


class ChatStream:
    """A reply that arrives piece by piece.

    Iterate over it to receive the text as it is generated. Once the iteration has
    ended, ``result`` holds the complete reply with token counts and timing. A stream
    can be read once. If you stop early, call ``close()`` (or use it in a ``with``
    block) to release the connection.
    """

    def __init__(self, client: LLMClient, payload: dict[str, Any]) -> None:
        self._client = client
        self._payload = payload
        self._result: ChatResult | None = None
        self._pieces = self._read()

    def __iter__(self) -> Iterator[str]:
        return self._pieces

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """Stop reading and release the connection. Safe to call more than once."""
        self._pieces.close()

    @property
    def result(self) -> ChatResult:
        """The complete reply. Available only after the stream has been read to its end."""
        if self._result is None:
            raise RuntimeError("The stream has not been read to its end yet.")
        return self._result

    def _read(self) -> Iterator[str]:
        started = time.perf_counter()
        parts: list[str] = []
        model: str = self._payload["model"]
        finish_reason: str | None = None
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        events = 0

        # One try block covers every point where the connection can fail: while
        # connecting, while reading an error reply, and while lines are arriving.
        try:
            with self._client._http.stream(
                "POST", "chat/completions", json=self._payload
            ) as response:
                if response.status_code >= 400:
                    response.read()
                    raise _status_error(response)
                for line in response.iter_lines():
                    event = _parse_event(line)
                    if event is None:
                        continue
                    if event is _DONE:
                        break
                    events += 1
                    if "error" in event:
                        raise LLMResponseError(
                            f"The LLM server reported an error while answering: "
                            f"{_error_text(event['error'])}"
                        )
                    model = event.get("model") or model
                    usage = _as_dict(event.get("usage"))
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                    completion_tokens = usage.get("completion_tokens", completion_tokens)
                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    choice = _as_dict(choices[0])
                    finish_reason = choice.get("finish_reason") or finish_reason
                    piece = _as_dict(choice.get("delta")).get("content")
                    if piece:
                        parts.append(piece)
                        yield piece
        except httpx.RequestError as exc:
            raise self._client._transport_error(exc) from exc

        if events == 0:
            raise LLMResponseError("The LLM server closed the stream without sending any data.")
        self._result = ChatResult(
            text="".join(parts),
            model=model,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_s=time.perf_counter() - started,
        )


class LLMClient:
    """Client for an OpenAI-compatible chat server, configured from ``config.yaml``.

    Args:
        config: The ``llm`` section of the configuration.
        transport: For tests only. A fake ``httpx`` transport, so that no server is needed.

    Raises:
        LLMConfigError: ``base_url`` is not an http(s) address, it points at another
            machine while ``allow_remote`` is false, or ``api_key_env`` names an
            environment variable that is not set.
    """

    def __init__(self, config: LLMConfig, *, transport: httpx.BaseTransport | None = None) -> None:
        base_url = config.base_url.rstrip("/")
        parts = urlsplit(base_url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise LLMConfigError(f"llm.base_url is not an http(s) address: {config.base_url!r}")
        local = is_local_url(base_url)
        if not local and not config.allow_remote:
            raise LLMConfigError(
                f"The model server address ({base_url}) is not on this machine and "
                f"llm.allow_remote is false. Nothing was sent. Set allow_remote to true only "
                f"on purpose: whatever the model is asked (questions with the passages found "
                f"for them, or figure images) is then sent to that server."
            )

        headers: dict[str, str] = {}
        if config.api_key_env:
            api_key = os.environ.get(config.api_key_env)
            if not api_key:
                raise LLMConfigError(
                    f"llm.api_key_env names the environment variable {config.api_key_env!r}, "
                    f"which is not set."
                )
            headers["Authorization"] = f"Bearer {api_key}"

        self._config = config
        self._base_url = base_url
        self._http = httpx.Client(
            base_url=base_url,
            headers=headers,
            timeout=httpx.Timeout(
                connect=config.connect_timeout_s,
                read=config.request_timeout_s,
                write=config.request_timeout_s,
                pool=config.connect_timeout_s,
            ),
            transport=transport,
            # A local server is called directly: proxy settings in the environment
            # must not reroute requests that are meant to stay on this machine.
            trust_env=not local,
        )

    # -- housekeeping -----------------------------------------------------------------

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def answer_model(self) -> str:
        return self._config.answer_model

    @property
    def vision_model(self) -> str:
        return self._config.vision_model

    def close(self) -> None:
        """Release the connection. Safe to call more than once."""
        self._http.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- models -----------------------------------------------------------------------

    def list_models(self) -> list[str]:
        """Names of the models the server has. Also the cheapest way to see if it is up."""
        data = self._request_json("GET", "models")
        try:
            return [str(item["id"]) for item in data["data"]]
        except (KeyError, TypeError) as exc:
            raise LLMResponseError("The LLM server's model list could not be understood.") from exc

    def has_model(self, name: str) -> bool:
        """True if the server has ``name``. A name without a tag means ``name:latest``."""
        return model_is_listed(name, self.list_models())

    # -- chat -------------------------------------------------------------------------

    def chat(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> ChatResult:
        """Send a conversation and wait for the complete reply.

        Args:
            messages: The conversation so far, oldest first. Build the items with
                ``system_message``, ``user_message`` and ``image_message``.
            model: Model to use. Defaults to ``llm.answer_model``.
            temperature: Defaults to ``llm.temperature``.
            max_tokens: Longest reply allowed. Defaults to ``llm.max_tokens``.
            extra: Further OpenAI-style request fields, for example ``{"seed": 1}``.

        Raises:
            LLMConnectionError: The server cannot be reached.
            LLMTimeoutError: The server did not answer within ``llm.request_timeout_s``.
            LLMResponseError: The server returned an error or an unreadable reply.
        """
        payload = self._payload(messages, model, temperature, max_tokens, extra, stream=False)
        started = time.perf_counter()
        data = self._request_json("POST", "chat/completions", json=payload)
        return _chat_result(data, payload["model"], time.perf_counter() - started)

    def chat_stream(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> ChatStream:
        """Send a conversation and receive the reply piece by piece.

        Takes the same arguments as ``chat``. Nothing is sent until the returned stream
        is iterated, and errors are raised from that iteration.
        """
        payload = self._payload(messages, model, temperature, max_tokens, extra, stream=True)
        return ChatStream(self, payload)

    def describe_image(
        self,
        prompt: str,
        image: bytes | str | os.PathLike[str],
        *,
        mime_type: str | None = None,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        """Ask the vision model about one image.

        Args:
            prompt: What to say about the image.
            image: The image bytes, or the path of an image file.
            mime_type: Needed only when it cannot be told from the file name. Bytes are
                taken to be PNG unless stated otherwise.
            system: Optional instructions for the model.
            model: Defaults to ``llm.vision_model``.
            max_tokens: Longest reply allowed. Defaults to ``llm.max_tokens``.
        """
        if isinstance(image, bytes):
            data = image
            mime_type = mime_type or "image/png"
        else:
            path = Path(image)
            data = path.read_bytes()
            mime_type = mime_type or mimetypes.guess_type(path.name)[0] or "unknown"
        messages = [system_message(system)] if system else []
        messages.append(image_message(prompt, data, mime_type))
        return self.chat(messages, model=model or self._config.vision_model, max_tokens=max_tokens)

    # -- internals --------------------------------------------------------------------

    def _payload(
        self,
        messages: Sequence[Message],
        model: str | None,
        temperature: float | None,
        max_tokens: int | None,
        extra: Mapping[str, Any] | None,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        if not messages:
            raise ValueError("messages must not be empty.")
        payload: dict[str, Any] = {
            "model": model or self._config.answer_model,
            "messages": list(messages),
            "temperature": self._config.temperature if temperature is None else temperature,
            "max_tokens": self._config.max_tokens if max_tokens is None else max_tokens,
            "stream": stream,
        }
        if stream:
            # Ask for token counts at the end of the stream, for the query log.
            payload["stream_options"] = {"include_usage": True}
        if extra:
            clash = _RESERVED_PAYLOAD_KEYS.intersection(extra)
            if clash:
                raise ValueError(f"extra must not set: {', '.join(sorted(clash))}.")
            payload.update(extra)
        return payload

    def _request_json(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = self._http.request(method, path, **kwargs)
        except httpx.RequestError as exc:
            raise self._transport_error(exc) from exc
        if response.status_code >= 400:
            raise _status_error(response)
        try:
            data = response.json()
        except ValueError as exc:
            raise LLMResponseError("The LLM server's reply was not valid JSON.") from exc
        if not isinstance(data, dict):
            raise LLMResponseError("The LLM server's reply was not a JSON object.")
        return data

    def _transport_error(self, exc: httpx.RequestError) -> LLMError:
        if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
            return LLMConnectionError(
                f"Cannot reach the LLM server at {self._base_url}. Is it running? ({exc})"
            )
        if isinstance(exc, httpx.TimeoutException):
            return LLMTimeoutError(
                f"The LLM server at {self._base_url} did not answer within "
                f"{self._config.request_timeout_s:g} seconds. If the model is just slow, "
                f"raise llm.request_timeout_s in config.yaml."
            )
        return LLMConnectionError(
            f"The connection to the LLM server at {self._base_url} failed: "
            f"{type(exc).__name__}: {exc}"
        )


def _parse_event(line: str) -> Any:
    """Read one line of a server-sent-event stream.

    Returns the JSON object it carries, ``_DONE`` for the end marker, or ``None`` for a
    line with no data (blank separators, comments, keep-alives).
    """
    line = line.strip()
    if not line.startswith("data:"):
        return None
    data = line[len("data:") :].strip()
    if not data:
        return None
    if data == "[DONE]":
        return _DONE
    try:
        event = json.loads(data)
    except ValueError as exc:
        raise LLMResponseError(
            f"The LLM server sent a line that is not valid JSON: {data[:200]!r}"
        ) from exc
    if not isinstance(event, dict):
        raise LLMResponseError(f"The LLM server sent an unexpected line: {data[:200]!r}")
    return event


def _chat_result(data: dict[str, Any], requested_model: str, latency_s: float) -> ChatResult:
    try:
        choice = data["choices"][0]
        text = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMResponseError("The LLM server's reply contained no answer.") from exc
    usage = _as_dict(data.get("usage"))
    return ChatResult(
        text=text or "",
        model=data.get("model") or requested_model,
        finish_reason=choice.get("finish_reason"),
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        latency_s=latency_s,
    )


def _status_error(response: httpx.Response) -> LLMResponseError:
    """Build the error for an HTTP error status, using the server's own message if it sent one."""
    message = ""
    try:
        body = response.json()
    except ValueError:
        body = None
    if isinstance(body, dict) and "error" in body:
        message = _error_text(body["error"])
    if not message:
        message = response.text.strip()[:300] or response.reason_phrase
    return LLMResponseError(
        f"The LLM server answered HTTP {response.status_code}: {message}",
        status_code=response.status_code,
    )


def _error_text(error: Any) -> str:
    if isinstance(error, dict):
        return str(error.get("message") or error)
    return str(error)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _with_default_tag(name: str) -> str:
    return name if ":" in name else f"{name}:latest"
