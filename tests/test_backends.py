import json
from typing import Any

import httpx
import pytest

from llmgater.backends import AnthropicBackend, Backend, OpenAIBackend, ScriptedBackend
from llmgater.backends.abstract import read_retry_after
from llmgater.exceptions import BackendError
from llmgater.models import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    StreamDelta,
    TokenUsage,
    approx_tokens,
)

REQUEST = ChatRequest(
    [ChatMessage("system", "Answer in one line."), ChatMessage("user", "Good morning")]
)
OPENAI_REPLY = {
    "choices": [{"message": {"role": "assistant", "content": "Morning!"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 14, "completion_tokens": 2, "total_tokens": 16},
}


def mock_server(status: int = 200, **kwargs: Any) -> tuple[httpx.AsyncClient, list[httpx.Request]]:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(status, **kwargs)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), captured


def event_stream(*events: dict[str, Any] | str) -> bytes:
    lines = (event if isinstance(event, str) else json.dumps(event) for event in events)
    return "".join(f"data: {line}\n\n" for line in lines).encode()


async def gather(backend: Backend, request: ChatRequest = REQUEST) -> list[StreamDelta]:
    return [delta async for delta in backend.generate_stream(request)]


async def test_openai_generate() -> None:
    client, captured = mock_server(json=OPENAI_REPLY)
    backend = OpenAIBackend("gpt-5-mini", api_key="sk-unit", client=client)

    reply = await backend.generate(REQUEST)

    assert reply == ChatResponse("Morning!", "gpt-5-mini", "openai", TokenUsage(14, 2), "stop")
    assert captured[0].url == "https://api.openai.com/v1/chat/completions"
    assert captured[0].headers["authorization"] == "Bearer sk-unit"
    assert json.loads(captured[0].content) == {
        "model": "gpt-5-mini",
        "messages": [
            {"role": "system", "content": "Answer in one line."},
            {"role": "user", "content": "Good morning"},
        ],
    }


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ({}, {}),
        ({"temperature": 1.0}, {"temperature": 1.0}),
        (
            {"temperature": 0.2, "max_output_tokens": 128},
            {"temperature": 0.2, "max_completion_tokens": 128},
        ),
        ({"max_output_tokens": 16}, {"max_completion_tokens": 16}),
    ],
)
async def test_openai_only_sends_options_that_were_set(
    options: dict[str, Any], expected: dict[str, Any]
) -> None:
    client, captured = mock_server(json=OPENAI_REPLY)
    backend = OpenAIBackend("gpt-5-mini", api_key="sk-unit", client=client)

    await backend.generate(ChatRequest(REQUEST.messages, **options))

    body = json.loads(captured[0].content)
    keys = ("temperature", "max_tokens", "max_completion_tokens")
    assert {key: body[key] for key in keys if key in body} == expected


async def test_compatible_servers_get_plain_max_tokens() -> None:
    client, captured = mock_server(json=OPENAI_REPLY)
    backend = OpenAIBackend("Qwen/Qwen3-8B", base_url="http://localhost:8000/v1", client=client)

    await backend.generate(ChatRequest(REQUEST.messages, max_output_tokens=128))

    body = json.loads(captured[0].content)
    assert body["max_tokens"] == 128
    assert "max_completion_tokens" not in body


async def test_openai_stream() -> None:
    client, captured = mock_server(
        content=event_stream(
            {"choices": [{"delta": {"role": "assistant", "content": ""}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": "Good"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": " day"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "length"}]},
            {"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 2}},
            "[DONE]",
            {"choices": [{"delta": {"content": "ignored after done"}}]},
        )
    )
    backend = OpenAIBackend("gpt-5-mini", api_key="sk-unit", client=client)

    deltas = await gather(backend)

    assert deltas == [
        StreamDelta("Good"),
        StreamDelta(" day"),
        StreamDelta(stop_reason="length"),
        StreamDelta(usage=TokenUsage(11, 2)),
    ]
    body = json.loads(captured[0].content)
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}


async def test_openai_env_key_stays_with_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    client, captured = mock_server(json=OPENAI_REPLY)

    await OpenAIBackend("gpt-5-mini", client=client).generate(REQUEST)
    await OpenAIBackend("llama3.2", base_url="http://localhost:11434/v1/", client=client).generate(
        REQUEST
    )

    assert captured[0].headers["authorization"] == "Bearer sk-from-env"
    assert captured[1].url == "http://localhost:11434/v1/chat/completions"
    assert "authorization" not in captured[1].headers


async def test_anthropic_generate() -> None:
    client, captured = mock_server(
        json={
            "content": [
                {"type": "thinking", "thinking": "", "signature": "sig"},
                {"type": "text", "text": "Guten"},
                {"type": "text", "text": " Morgen"},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 18, "output_tokens": 3},
        }
    )
    backend = AnthropicBackend("claude-haiku-5-5", api_key="sk-ant-unit", client=client)
    system = [ChatMessage("system", "Answer in one line."), ChatMessage("system", "Use German.")]

    reply = await backend.generate(
        ChatRequest([*system, ChatMessage("user", "Good morning")], temperature=0.0)
    )

    assert reply == ChatResponse(
        "Guten Morgen", "claude-haiku-5-5", "anthropic", TokenUsage(18, 3), "end_turn"
    )
    assert captured[0].url == "https://api.anthropic.com/v1/messages"
    assert captured[0].headers["x-api-key"] == "sk-ant-unit"
    assert captured[0].headers["anthropic-version"] == "2023-06-01"
    assert json.loads(captured[0].content) == {
        "model": "claude-haiku-5-5",
        "max_tokens": 1024,
        "temperature": 0.0,
        "system": "Answer in one line.\n\nUse German.",
        "messages": [{"role": "user", "content": "Good morning"}],
    }


async def test_anthropic_request_overrides_default_max_tokens() -> None:
    client, captured = mock_server(
        json={"content": [], "usage": {"input_tokens": 1, "output_tokens": 0}}
    )
    backend = AnthropicBackend(
        "claude-haiku-5-5", api_key="sk-ant-unit", default_max_tokens=256, client=client
    )

    await backend.generate(REQUEST)
    await backend.generate(ChatRequest(REQUEST.messages, max_output_tokens=32))

    assert [json.loads(request.content)["max_tokens"] for request in captured] == [256, 32]


async def test_anthropic_stream() -> None:
    client, captured = mock_server(
        content=event_stream(
            {"type": "message_start", "message": {"usage": {"input_tokens": 21}}},
            {"type": "ping"},
            {"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "..."}},
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Good"}},
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": " day"}},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "max_tokens"},
                "usage": {"output_tokens": 9},
            },
            {"type": "message_stop"},
        )
    )
    backend = AnthropicBackend("claude-haiku-5-5", api_key="sk-ant-unit", client=client)

    deltas = await gather(backend)

    assert deltas == [
        StreamDelta("Good"),
        StreamDelta(" day"),
        StreamDelta(usage=TokenUsage(21, 9), stop_reason="max_tokens"),
    ]
    assert json.loads(captured[0].content)["stream"] is True


@pytest.mark.parametrize(
    ("kind", "transient"),
    [("overloaded_error", True), ("rate_limit_error", True), ("invalid_request_error", False)],
)
async def test_anthropic_error_event_mid_stream(kind: str, transient: bool) -> None:
    client, _ = mock_server(
        content=event_stream(
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Go"}},
            {"type": "error", "error": {"type": kind, "message": "stream cut"}},
        )
    )
    backend = AnthropicBackend("claude-haiku-5-5", api_key="sk-ant-unit", client=client)
    deltas: list[StreamDelta] = []

    with pytest.raises(BackendError, match="stream cut") as caught:
        async for delta in backend.generate_stream(REQUEST):
            deltas.append(delta)

    assert deltas == [StreamDelta("Go")]
    assert caught.value.transient is transient


def test_anthropic_refuses_to_start_without_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        AnthropicBackend("claude-haiku-5-5")


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    ("status", "headers", "transient", "wait_hint"),
    [
        (429, {"Retry-After": "7"}, True, 7.0),
        (503, {"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}, True, None),
        (502, {}, True, None),
        (409, {"Retry-After": "-1"}, True, None),
        (403, {}, False, None),
        (400, {}, False, None),
    ],
)
async def test_http_errors_are_translated(
    status: int,
    headers: dict[str, str],
    transient: bool,
    wait_hint: float | None,
    streaming: bool,
) -> None:
    client, _ = mock_server(
        status, headers=headers, json={"error": {"type": "error", "message": "denied"}}
    )
    backend = OpenAIBackend("gpt-5-mini", api_key="sk-unit", client=client)

    with pytest.raises(BackendError) as caught:
        await (gather(backend) if streaming else backend.generate(REQUEST))

    err = caught.value
    assert str(err) == "[openai] denied"
    assert err.status == status
    assert (err.transient, err.wait_hint) == (transient, wait_hint)


async def test_error_without_json_body_falls_back_to_status_line() -> None:
    client, _ = mock_server(504, content=b"upstream timeout")
    backend = OpenAIBackend("gpt-5-mini", api_key="sk-unit", client=client)

    with pytest.raises(BackendError, match="HTTP 504 Gateway Timeout") as caught:
        await backend.generate(REQUEST)

    assert caught.value.transient


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "failure", [httpx.ConnectError("connection refused"), httpx.WriteTimeout("")]
)
async def test_network_failures_are_transient(failure: httpx.HTTPError, streaming: bool) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise failure

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend = OpenAIBackend("gpt-5-mini", api_key="sk-unit", client=client)

    with pytest.raises(BackendError) as caught:
        await (gather(backend) if streaming else backend.generate(REQUEST))

    assert caught.value.transient
    assert caught.value.status is None


@pytest.mark.parametrize(
    "body",
    [
        {"content": b"<html>502 Bad Gateway</html>"},
        {"json": [1, 2, 3]},
        {"json": {"choices": []}},
        {"json": {"choices": [{"message": None}]}},
        {"json": {"choices": [{"text": "legacy completions shape"}]}},
    ],
)
async def test_unexpected_shapes_raise_backend_error(body: dict[str, Any]) -> None:
    client, _ = mock_server(**body)
    backend = OpenAIBackend("gpt-5-mini", api_key="sk-unit", client=client)

    with pytest.raises(BackendError, match="unexpected reply format"):
        await backend.generate(REQUEST)


async def test_error_envelope_inside_a_200_is_still_an_error() -> None:
    client, _ = mock_server(json={"error": {"type": "server_error", "message": "try later"}})
    backend = OpenAIBackend("gpt-5-mini", api_key="sk-unit", client=client)

    with pytest.raises(BackendError, match="try later") as caught:
        await backend.generate(REQUEST)

    assert caught.value.transient


@pytest.mark.parametrize(
    ("header", "seconds"),
    [("3.5", 3.5), ("0", 0.0), ("inf", None), ("nan", None), ("-2", None), ("soon", None)],
)
def test_read_retry_after(header: str, seconds: float | None) -> None:
    assert read_retry_after(header) == seconds
    assert read_retry_after(None) is None


async def test_scripted_backend_plays_its_script_in_order() -> None:
    hiccup = BackendError("hiccup", backend="scripted", transient=True)
    backend = ScriptedBackend(["one", hiccup, "two", "final"])

    assert (await backend.generate(REQUEST)).text == "one"
    with pytest.raises(BackendError) as caught:
        await backend.generate(REQUEST)
    assert caught.value is hiccup
    assert (await backend.generate(REQUEST)).text == "two"
    assert [(await backend.generate(REQUEST)).text for _ in range(3)] == ["final"] * 3
    assert len(backend.received) == 6


def test_scripted_backend_needs_a_script() -> None:
    with pytest.raises(ValueError, match="non-empty script"):
        ScriptedBackend([])


async def test_scripted_stream_reassembles_the_exact_text() -> None:
    text = " Rain,  then sun\n and wind "

    deltas = await gather(ScriptedBackend(text))

    assert [delta.text for delta in deltas[:-1]] == [" Rain,  ", "then ", "sun\n ", "and ", "wind "]
    assert "".join(delta.text for delta in deltas) == text
    prompt = sum(approx_tokens(message.content) for message in REQUEST.messages)
    assert deltas[-1] == StreamDelta(
        usage=TokenUsage(prompt, approx_tokens(text)), stop_reason="stop"
    )


async def test_close_leaves_injected_clients_open() -> None:
    injected = httpx.AsyncClient()
    shared = OpenAIBackend("gpt-5-mini", api_key="sk-unit", client=injected)
    owned = AnthropicBackend("claude-haiku-5-5", api_key="sk-ant-unit")

    await shared.close()
    await owned.close()

    assert not injected.is_closed
    assert owned.http.is_closed
    await injected.aclose()


def test_backend_repr() -> None:
    assert (
        repr(ScriptedBackend(name="demo", model="m-1"))
        == "ScriptedBackend(name='demo', model='m-1')"
    )
