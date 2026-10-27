import os
from collections.abc import AsyncGenerator
from contextlib import aclosing
from typing import Any

import httpx

from llmgater.backends.abstract import HTTPBackend
from llmgater.models import ChatRequest, ChatResponse, StreamDelta, TokenUsage

DEFAULT_ANTHROPIC_URL = "https://api.anthropic.com/v1"
ANTHROPIC_API_VERSION = "2023-06-01"


class AnthropicBackend(HTTPBackend):
    """Anthropic Messages API."""

    api_key: str

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        name: str = "anthropic",
        default_max_tokens: int = 1024,
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("no Anthropic API key: pass api_key= or set ANTHROPIC_API_KEY")
        super().__init__(
            model,
            name=name,
            api_key=key,
            base_url=base_url or DEFAULT_ANTHROPIC_URL,
            timeout=timeout,
            client=client,
        )
        self.default_max_tokens = default_max_tokens

    def auth_headers(self) -> dict[str, str]:
        """Anthropic wants an x-api-key header plus a pinned API version."""
        return {"x-api-key": self.api_key, "anthropic-version": ANTHROPIC_API_VERSION}

    def build_body(self, request: ChatRequest) -> dict[str, Any]:
        """Translate a ChatRequest into a Messages body; system turns move to the top level."""
        system_parts = [turn.content for turn in request.messages if turn.role == "system"]
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": request.max_output_tokens or self.default_max_tokens,
            "messages": [
                {"role": turn.role, "content": turn.content}
                for turn in request.messages
                if turn.role != "system"
            ],
        }
        if system_parts:
            body["system"] = "\n\n".join(system_parts)
        if request.temperature is not None:
            body["temperature"] = request.temperature
        return body

    async def generate(self, request: ChatRequest) -> ChatResponse:
        """Send one non-streaming Messages request."""
        data = await self.post_json("/messages", self.build_body(request))
        with self.guard_parse():
            text = "".join(block["text"] for block in data["content"] if block["type"] == "text")
            counts = data["usage"]
            return ChatResponse(
                text=text,
                model=self.model,
                backend=self.name,
                usage=TokenUsage(int(counts["input_tokens"]), int(counts["output_tokens"])),
                stop_reason=data.get("stop_reason"),
            )

    async def generate_stream(self, request: ChatRequest) -> AsyncGenerator[StreamDelta, None]:
        """Yield text deltas, then one closing delta with usage and the stop reason."""
        usage = TokenUsage()
        stop_reason: str | None = None
        body = {**self.build_body(request), "stream": True}
        async with aclosing(self.sse_events("/messages", body)) as events:
            async for event in events:
                piece: StreamDelta | None = None
                with self.guard_parse():
                    kind = event["type"]
                    if kind == "message_start":
                        usage.tokens_in = int(event["message"]["usage"]["input_tokens"])
                    elif kind == "content_block_delta" and event["delta"]["type"] == "text_delta":
                        piece = StreamDelta(event["delta"]["text"])
                    elif kind == "message_delta":
                        usage.tokens_out = int(event["usage"]["output_tokens"])
                        stop_reason = event["delta"].get("stop_reason")
                if piece is not None:
                    yield piece
        yield StreamDelta(usage=usage, stop_reason=stop_reason)
