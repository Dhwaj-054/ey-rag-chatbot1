"""Tests for the LLM client.

No server is needed: the HTTP layer is replaced with ``httpx.MockTransport``, so each
test sees exactly the request the client would send to Ollama and decides the reply.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from docqa.config import LLMConfig
from docqa.llm import (
    LLMClient,
    LLMConfigError,
    LLMConnectionError,
    LLMResponseError,
    LLMTimeoutError,
    image_message,
    is_local_url,
    model_is_listed,
    system_message,
    user_message,
)

Handler = Callable[[httpx.Request], httpx.Response]

CHAT_REPLY = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "model": "answer-model",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello there."},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
}

PNG_BYTES = b"\x89PNG\r\n\x1a\n-not-a-real-image-"


def make_config(**overrides: Any) -> LLMConfig:
    values: dict[str, Any] = {
        "base_url": "http://localhost:11434/v1",
        "allow_remote": False,
        "api_key_env": None,
        "answer_model": "answer-model",
        "vision_model": "vision-model",
        "temperature": 0.0,
        "max_tokens": 256,
        "connect_timeout_s": 1,
        "request_timeout_s": 5,
    }
    values.update(overrides)
    return LLMConfig(**values)


class FakeServer:
    """Stands in for the LLM server: records every request and answers with ``handler``."""

    def __init__(self, handler: Handler) -> None:
        self.handler = handler
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.handler(request)

    @property
    def sent(self) -> dict[str, Any]:
        """The JSON body of the most recent request."""
        return json.loads(self.requests[-1].content)


def json_reply(data: Any, status: int = 200) -> Handler:
    return lambda request: httpx.Response(status, json=data)


def sse_reply(*lines: str) -> Handler:
    body = "".join(f"{line}\n\n" for line in lines).encode("utf-8")
    headers = {"content-type": "text/event-stream"}
    return lambda request: httpx.Response(200, headers=headers, content=body)


def failing_with(error: type[httpx.RequestError]) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        raise error("simulated failure", request=request)

    return handler


def chunk(content: str | None = None, finish: str | None = None) -> str:
    """One streamed line in the format Ollama sends."""
    delta = {} if content is None else {"content": content}
    event = {
        "model": "answer-model",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return "data: " + json.dumps(event)


USAGE_LINE = "data: " + json.dumps(
    {"model": "answer-model", "choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 2}}
)
DONE_LINE = "data: [DONE]"


def make_client(handler: Handler, **overrides: Any) -> tuple[LLMClient, FakeServer]:
    server = FakeServer(handler)
    client = LLMClient(make_config(**overrides), transport=httpx.MockTransport(server))
    return client, server


QUESTION = [system_message("Answer briefly."), user_message("Hello")]


# -- chat -----------------------------------------------------------------------------


def test_chat_sends_an_openai_style_request():
    client, server = make_client(json_reply(CHAT_REPLY))

    client.chat(QUESTION)

    request = server.requests[0]
    assert request.method == "POST"
    assert str(request.url) == "http://localhost:11434/v1/chat/completions"
    assert server.sent == {
        "model": "answer-model",
        "messages": [
            {"role": "system", "content": "Answer briefly."},
            {"role": "user", "content": "Hello"},
        ],
        "temperature": 0.0,
        "max_tokens": 256,
        "stream": False,
    }


def test_chat_returns_text_model_and_token_counts():
    client, _ = make_client(json_reply(CHAT_REPLY))

    reply = client.chat(QUESTION)

    assert reply.text == "Hello there."
    assert reply.model == "answer-model"
    assert reply.finish_reason == "stop"
    assert reply.prompt_tokens == 12
    assert reply.completion_tokens == 3
    assert reply.latency_s >= 0


def test_chat_arguments_override_the_configured_defaults():
    client, server = make_client(json_reply(CHAT_REPLY))

    client.chat(QUESTION, model="other-model", temperature=0.7, max_tokens=50, extra={"seed": 7})

    assert server.sent["model"] == "other-model"
    assert server.sent["temperature"] == 0.7
    assert server.sent["max_tokens"] == 50
    assert server.sent["seed"] == 7


def test_chat_copes_with_a_reply_that_has_no_usage_or_model():
    reply = {"choices": [{"message": {"role": "assistant", "content": None}}]}
    client, _ = make_client(json_reply(reply))

    result = client.chat(QUESTION)

    assert result.text == ""
    assert result.model == "answer-model"  # falls back to the model that was asked for
    assert result.prompt_tokens is None
    assert result.finish_reason is None


@pytest.mark.parametrize("reserved", ["model", "messages", "stream"])
def test_extra_cannot_replace_the_core_fields(reserved):
    client, server = make_client(json_reply(CHAT_REPLY))

    with pytest.raises(ValueError, match=reserved):
        client.chat(QUESTION, extra={reserved: "x"})
    assert server.requests == []


def test_chat_needs_at_least_one_message():
    client, server = make_client(json_reply(CHAT_REPLY))

    with pytest.raises(ValueError, match="must not be empty"):
        client.chat([])
    assert server.requests == []


def test_a_trailing_slash_in_base_url_is_harmless():
    client, server = make_client(json_reply(CHAT_REPLY), base_url="http://localhost:11434/v1/")

    client.chat(QUESTION)

    assert str(server.requests[0].url) == "http://localhost:11434/v1/chat/completions"


# -- streaming ------------------------------------------------------------------------


def test_stream_yields_text_as_it_arrives_and_then_the_whole_reply():
    lines = [chunk(""), chunk("Hel"), ": keep-alive", chunk("lo"), chunk(finish="stop")]
    client, server = make_client(sse_reply(*lines, USAGE_LINE, DONE_LINE))

    stream = client.chat_stream(QUESTION)
    pieces = list(stream)

    assert pieces == ["Hel", "lo"]
    result = stream.result
    assert result.text == "Hello"
    assert result.model == "answer-model"
    assert result.finish_reason == "stop"
    assert result.prompt_tokens == 12
    assert result.completion_tokens == 2
    assert result.latency_s >= 0
    assert server.sent["stream"] is True
    assert server.sent["stream_options"] == {"include_usage": True}


def test_stream_sends_nothing_until_it_is_read():
    client, server = make_client(sse_reply(chunk("Hi"), DONE_LINE))

    stream = client.chat_stream(QUESTION)
    assert server.requests == []

    assert list(stream) == ["Hi"]
    assert len(server.requests) == 1


def test_stream_ignores_anything_after_the_end_marker():
    client, _ = make_client(sse_reply(chunk("Hi"), DONE_LINE, chunk("ignored")))

    assert list(client.chat_stream(QUESTION)) == ["Hi"]


def test_stream_without_an_end_marker_still_completes():
    client, _ = make_client(sse_reply(chunk("Hi"), chunk(finish="stop")))

    stream = client.chat_stream(QUESTION)

    assert list(stream) == ["Hi"]
    assert stream.result.text == "Hi"


def test_stream_result_is_not_available_until_the_end():
    client, _ = make_client(sse_reply(chunk("Hel"), chunk("lo"), DONE_LINE))

    stream = client.chat_stream(QUESTION)
    with pytest.raises(RuntimeError, match="not been read"):
        stream.result  # noqa: B018
    next(iter(stream))
    with pytest.raises(RuntimeError, match="not been read"):
        stream.result  # noqa: B018


def test_stream_can_be_closed_early():
    client, _ = make_client(sse_reply(chunk("Hel"), chunk("lo"), DONE_LINE))

    with client.chat_stream(QUESTION) as stream:
        assert next(iter(stream)) == "Hel"
    stream.close()  # a second close is harmless

    assert list(stream) == []
    with pytest.raises(RuntimeError):
        stream.result  # noqa: B018


def test_stream_reports_an_error_status():
    error = {"error": {"message": "model 'answer-model' not found", "type": "not_found_error"}}
    client, _ = make_client(json_reply(error, status=404))

    with pytest.raises(LLMResponseError, match="not found") as raised:
        list(client.chat_stream(QUESTION))
    assert raised.value.status_code == 404


def test_stream_reports_an_error_sent_part_way_through():
    client, _ = make_client(
        sse_reply(chunk("Hel"), 'data: {"error": {"message": "out of memory"}}')
    )

    pieces = []
    with pytest.raises(LLMResponseError, match="out of memory"):
        for piece in client.chat_stream(QUESTION):
            pieces.append(piece)
    assert pieces == ["Hel"]


def test_stream_that_sends_nothing_is_an_error():
    client, _ = make_client(lambda request: httpx.Response(200, content=b""))

    with pytest.raises(LLMResponseError, match="without sending any data"):
        list(client.chat_stream(QUESTION))


def test_stream_line_that_is_not_json_is_an_error():
    client, _ = make_client(sse_reply("data: {not json"))

    with pytest.raises(LLMResponseError, match="not valid JSON"):
        list(client.chat_stream(QUESTION))


def test_stream_reports_a_connection_that_breaks_part_way_through():
    class BreaksAfterFirstLine(httpx.SyncByteStream):
        def __iter__(self):
            yield (chunk("Hel") + "\n\n").encode("utf-8")
            raise httpx.ReadError("connection reset")

    client, _ = make_client(lambda request: httpx.Response(200, stream=BreaksAfterFirstLine()))

    pieces = []
    with pytest.raises(LLMConnectionError, match="connection reset"):
        for piece in client.chat_stream(QUESTION):
            pieces.append(piece)
    assert pieces == ["Hel"]


# -- failures -------------------------------------------------------------------------


@pytest.mark.parametrize("error", [httpx.ConnectError, httpx.ConnectTimeout])
@pytest.mark.parametrize("streaming", [False, True])
def test_server_that_is_not_running_is_a_connection_error(error, streaming):
    client, _ = make_client(failing_with(error))

    with pytest.raises(LLMConnectionError, match="Cannot reach the LLM server") as raised:
        if streaming:
            list(client.chat_stream(QUESTION))
        else:
            client.chat(QUESTION)
    assert "http://localhost:11434/v1" in str(raised.value)


@pytest.mark.parametrize("streaming", [False, True])
def test_slow_server_is_a_timeout_error_that_names_the_setting(streaming):
    client, _ = make_client(failing_with(httpx.ReadTimeout))

    with pytest.raises(LLMTimeoutError, match="request_timeout_s") as raised:
        if streaming:
            list(client.chat_stream(QUESTION))
        else:
            client.chat(QUESTION)
    assert "5 seconds" in str(raised.value)


def test_other_transport_failures_are_connection_errors():
    client, _ = make_client(failing_with(httpx.RemoteProtocolError))

    with pytest.raises(LLMConnectionError, match="RemoteProtocolError"):
        client.chat(QUESTION)


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ({"error": {"message": "model is loading"}}, "model is loading"),
        ({"error": "plain string error"}, "plain string error"),
    ],
)
def test_error_status_carries_the_servers_message(body, expected):
    client, _ = make_client(json_reply(body, status=500))

    with pytest.raises(LLMResponseError, match=expected) as raised:
        client.chat(QUESTION)
    assert raised.value.status_code == 500
    assert "HTTP 500" in str(raised.value)


def test_error_status_without_json_uses_the_body_text():
    client, _ = make_client(lambda request: httpx.Response(502, text="Bad Gateway from proxy"))

    with pytest.raises(LLMResponseError, match="Bad Gateway from proxy") as raised:
        client.chat(QUESTION)
    assert raised.value.status_code == 502


def test_reply_that_is_not_json_is_an_error():
    client, _ = make_client(lambda request: httpx.Response(200, text="<html>hello</html>"))

    with pytest.raises(LLMResponseError, match="not valid JSON"):
        client.chat(QUESTION)


@pytest.mark.parametrize("body", [{"choices": []}, {"object": "chat.completion"}, ["a", "list"]])
def test_reply_without_an_answer_is_an_error(body):
    client, _ = make_client(json_reply(body))

    with pytest.raises(LLMResponseError):
        client.chat(QUESTION)


# -- models ---------------------------------------------------------------------------

MODEL_LIST = {
    "object": "list",
    "data": [
        {"id": "qwen2.5:7b", "object": "model", "owned_by": "library"},
        {"id": "llama3:latest", "object": "model", "owned_by": "library"},
    ],
}


def test_list_models_asks_the_models_endpoint():
    client, server = make_client(json_reply(MODEL_LIST))

    assert client.list_models() == ["qwen2.5:7b", "llama3:latest"]
    assert server.requests[0].method == "GET"
    assert str(server.requests[0].url) == "http://localhost:11434/v1/models"


def test_has_model():
    client, _ = make_client(json_reply(MODEL_LIST))

    assert client.has_model("qwen2.5:7b")
    assert client.has_model("llama3")  # no tag means :latest
    assert not client.has_model("qwen2.5")  # :latest is not installed, only :7b
    assert not client.has_model("qwen2.5vl:7b")


def test_model_list_that_cannot_be_understood_is_an_error():
    client, _ = make_client(json_reply({"models": []}))

    with pytest.raises(LLMResponseError, match="model list"):
        client.list_models()


@pytest.mark.parametrize(
    ("name", "available", "expected"),
    [
        ("qwen2.5:7b", ["qwen2.5:7b"], True),
        ("qwen2.5:7b", ["qwen2.5:14b"], False),
        ("llama3", ["llama3:latest"], True),
        ("llama3:latest", ["llama3"], True),
        ("Qwen/Qwen2.5-7B-Instruct", ["Qwen/Qwen2.5-7B-Instruct"], True),  # a vLLM-style name
        ("anything", [], False),
    ],
)
def test_model_is_listed(name, available, expected):
    assert model_is_listed(name, available) is expected


# -- images ---------------------------------------------------------------------------


def test_image_message_carries_the_image_as_a_base64_data_url():
    message = image_message("Describe this figure.", PNG_BYTES)

    assert message["role"] == "user"
    text_part, image_part = message["content"]
    assert text_part == {"type": "text", "text": "Describe this figure."}
    assert image_part["type"] == "image_url"
    prefix = "data:image/png;base64,"
    url = image_part["image_url"]["url"]
    assert url.startswith(prefix)
    assert base64.b64decode(url[len(prefix) :]) == PNG_BYTES


def test_image_message_rejects_unsupported_types_and_empty_images():
    with pytest.raises(ValueError, match="Unsupported image type"):
        image_message("Describe.", PNG_BYTES, mime_type="image/tiff")
    with pytest.raises(ValueError, match="empty"):
        image_message("Describe.", b"")


def test_describe_image_uses_the_vision_model():
    client, server = make_client(json_reply({**CHAT_REPLY, "model": "vision-model"}))

    reply = client.describe_image("Describe this figure.", PNG_BYTES, system="Be exact.")

    assert reply.model == "vision-model"
    assert server.sent["model"] == "vision-model"
    system, user = server.sent["messages"]
    assert system == {"role": "system", "content": "Be exact."}
    assert user["content"][0]["text"] == "Describe this figure."
    assert user["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_describe_image_reads_a_file_and_takes_the_type_from_its_name(tmp_path):
    path = tmp_path / "figure.jpg"
    path.write_bytes(b"jpeg-bytes")
    client, server = make_client(json_reply(CHAT_REPLY))

    client.describe_image("Describe.", path)

    url = server.sent["messages"][0]["content"][1]["image_url"]["url"]
    prefix = "data:image/jpeg;base64,"
    assert url.startswith(prefix)
    assert base64.b64decode(url[len(prefix) :]) == b"jpeg-bytes"


def test_describe_image_refuses_a_file_type_the_server_cannot_read(tmp_path):
    path = tmp_path / "figure.bmp"
    path.write_bytes(b"bmp-bytes")
    client, server = make_client(json_reply(CHAT_REPLY))

    with pytest.raises(ValueError, match="Unsupported image type"):
        client.describe_image("Describe.", path)
    assert server.requests == []


# -- nothing leaves the machine -------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://localhost:11434/v1", True),
        ("http://LOCALHOST:11434/v1", True),
        ("http://127.0.0.1:11434/v1", True),
        ("http://127.0.0.2:8000/v1", True),
        ("http://[::1]:11434/v1", True),
        ("http://192.168.1.10:11434/v1", False),
        ("https://api.example.com/v1", False),
        ("http://localhost.example.com/v1", False),
        ("localhost:11434", False),
        ("", False),
    ],
)
def test_is_local_url(url, expected):
    assert is_local_url(url) is expected


def test_remote_server_is_refused_unless_allowed():
    server = FakeServer(json_reply(CHAT_REPLY))
    transport = httpx.MockTransport(server)

    with pytest.raises(LLMConfigError, match="allow_remote"):
        LLMClient(make_config(base_url="https://llm.example.com/v1"), transport=transport)
    assert server.requests == []


def test_remote_server_works_when_allowed():
    """The later move to a GPU server is a config change."""
    client, server = make_client(
        json_reply(CHAT_REPLY), base_url="https://gpu-box.internal:8000/v1", allow_remote=True
    )

    client.chat(QUESTION)

    assert str(server.requests[0].url) == "https://gpu-box.internal:8000/v1/chat/completions"


@pytest.mark.parametrize("url", ["localhost:11434/v1", "ftp://localhost/v1", "http://", "nonsense"])
def test_base_url_must_be_an_http_address(url):
    with pytest.raises(LLMConfigError, match="not an http"):
        LLMClient(make_config(base_url=url, allow_remote=True))


def test_local_server_is_called_directly_not_through_a_proxy():
    local, _ = make_client(json_reply(CHAT_REPLY))
    remote, _ = make_client(
        json_reply(CHAT_REPLY), base_url="https://gpu-box.internal:8000/v1", allow_remote=True
    )

    assert local._http.trust_env is False  # proxy settings in the environment are ignored
    assert remote._http.trust_env is True


def test_no_authorization_header_by_default():
    client, server = make_client(json_reply(CHAT_REPLY))

    client.chat(QUESTION)

    assert "authorization" not in server.requests[0].headers


def test_api_key_is_read_from_the_named_environment_variable(monkeypatch):
    monkeypatch.setenv("DOCQA_TEST_KEY", "secret-token")
    client, server = make_client(json_reply(CHAT_REPLY), api_key_env="DOCQA_TEST_KEY")

    client.chat(QUESTION)

    assert server.requests[0].headers["authorization"] == "Bearer secret-token"


def test_missing_api_key_variable_is_reported(monkeypatch):
    monkeypatch.delenv("DOCQA_TEST_KEY", raising=False)

    with pytest.raises(LLMConfigError, match="DOCQA_TEST_KEY"):
        LLMClient(make_config(api_key_env="DOCQA_TEST_KEY"))


def test_client_can_be_used_as_a_context_manager():
    server = FakeServer(json_reply(CHAT_REPLY))

    with LLMClient(make_config(), transport=httpx.MockTransport(server)) as client:
        assert client.base_url == "http://localhost:11434/v1"
        assert client.answer_model == "answer-model"
        assert client.vision_model == "vision-model"
        client.chat(QUESTION)
    client.close()  # a second close is harmless


def test_a_stream_that_is_stopped_early_says_what_had_arrived():
    """An answer stopped at its time limit keeps the words the model had written by then."""
    client, _ = make_client(sse_reply(chunk("The fee "), chunk("is 0.50%"), chunk(" of the amount."), DONE_LINE))
    stream = client.chat_stream(QUESTION)
    assert stream.so_far().text == ""  # nothing has been asked for yet

    pieces = iter(stream)
    next(pieces)
    next(pieces)
    stream.close()
    partial = stream.so_far()

    assert partial.text == "The fee is 0.50%"
    assert partial.finish_reason is None
    assert (partial.prompt_tokens, partial.completion_tokens) == (None, None)  # sent only with the end
    assert partial.latency_s >= 0


def test_a_stream_read_to_its_end_gives_the_same_reply_either_way():
    client, _ = make_client(sse_reply(chunk("Hel"), chunk("lo"), DONE_LINE))
    stream = client.chat_stream(QUESTION)

    assert list(stream) == ["Hel", "lo"]
    assert stream.so_far() is stream.result
