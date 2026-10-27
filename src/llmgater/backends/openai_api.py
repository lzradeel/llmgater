import os
from collections.abc import AsyncGenerator
from contextlib import aclosing
from typing import Any

import httpx

from llmgater.backends.abstract import HTTPBackend
from llmgater.models import ChatRequest, ChatResponse, StreamDelta, TokenUsage

DEFAULT_OPENAI_URL = "https://api.openai.com/v1"


class OpenAIBackend(HTTPBackend):
    """OpenAI Chat Completions, or any OpenAI-compatible server reached through base_url."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        name: str = "openai",
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not base_url:
            # The env key belongs to OpenAI and must never be sent to a third-party host.
            base_url = DEFAULT_OPENAI_URL
            api_key = api_key or os.environ.get("OPENAI_API_KEY")
        super().__init__(
            model, name=name, api_key=api_key, base_url=base_url, timeout=timeout, client=client
        )
        # OpenAI itself expects max_completion_tokens; compatible servers still read max_tokens.
        is_openai = self.base_url == DEFAULT_OPENAI_URL
        self.token_limit_field = "max_completion_tokens" if is_openai else "max_tokens"

    def build_body(self, request: ChatRequest) -> dict[str, Any]:
        """Translate a ChatRequest into a Chat Completions body."""
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": turn.role, "content": turn.content} for turn in request.messages],
        }
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            body[self.token_limit_field] = request.max_output_tokens
        return body

    async def generate(self, request: ChatRequest) -> ChatResponse:
        """Send one non-streaming chat completion."""
        data = await self.post_json("/chat/completions", self.build_body(request))
        with self.guard_parse():
            choice = data["choices"][0]
            return ChatResponse(
                text=choice["message"]["content"] or "",
                model=self.model,
                backend=self.name,
                usage=read_usage(data.get("usage") or {}),
                stop_reason=choice.get("finish_reason"),
            )

    async def generate_stream(self, request: ChatRequest) -> AsyncGenerator[StreamDelta, None]:
        """Yield deltas; usage arrives in a last event whose choices list is empty."""
        body = {
            **self.build_body(request),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        async with aclosing(self.sse_events("/chat/completions", body)) as events:
            async for event in events:
                with self.guard_parse():
                    choice = (event.get("choices") or [{}])[0]
                    piece = StreamDelta(
                        text=(choice.get("delta") or {}).get("content") or "",
                        usage=read_usage(event["usage"]) if event.get("usage") else None,
                        stop_reason=choice.get("finish_reason"),
                    )
                if piece.text or piece.usage or piece.stop_reason:
                    yield piece


def read_usage(raw: dict[str, Any]) -> TokenUsage:
    """Map OpenAI's prompt and completion token counts onto TokenUsage."""
    return TokenUsage(int(raw.get("prompt_tokens") or 0), int(raw.get("completion_tokens") or 0))
