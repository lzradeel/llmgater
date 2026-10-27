import asyncio
import re
from collections.abc import AsyncGenerator, Sequence

from llmgater.backends.abstract import Backend
from llmgater.models import ChatRequest, ChatResponse, StreamDelta, TokenUsage, approx_tokens

_WORD_WITH_SPACE = re.compile(r"\s*\S+\s*|\s+")


class ScriptedBackend(Backend):
    """Replays a fixed script of replies and errors; meant for tests and offline demos."""

    def __init__(
        self,
        script: str | Sequence[str | Exception] = "ok",
        *,
        name: str = "scripted",
        model: str = "scripted-1",
        latency: float = 0.0,
    ) -> None:
        self.script = [script] if isinstance(script, str) else list(script)
        if not self.script:
            raise ValueError("ScriptedBackend needs a non-empty script")
        self.name = name
        self.model = model
        self.latency = latency
        self.received: list[ChatRequest] = []

    async def generate(self, request: ChatRequest) -> ChatResponse:
        """Return the next scripted reply, or raise the next scripted error."""
        text = await self._next_entry(request)
        return ChatResponse(
            text=text,
            model=self.model,
            backend=self.name,
            usage=rough_usage(request, text),
            stop_reason="stop",
        )

    async def generate_stream(self, request: ChatRequest) -> AsyncGenerator[StreamDelta, None]:
        """Yield the next scripted reply one word at a time, followed by usage."""
        text = await self._next_entry(request)
        for piece in _WORD_WITH_SPACE.findall(text):
            yield StreamDelta(piece)
        yield StreamDelta(usage=rough_usage(request, text), stop_reason="stop")

    async def _next_entry(self, request: ChatRequest) -> str:
        """Record the request and take the next script entry; the last one repeats forever."""
        self.received.append(request)
        entry = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if self.latency > 0:
            await asyncio.sleep(self.latency)
        if isinstance(entry, Exception):
            raise entry
        return entry


def rough_usage(request: ChatRequest, text: str) -> TokenUsage:
    """Estimate usage for a request and reply without a real tokenizer."""
    prompt = sum(approx_tokens(turn.content) for turn in request.messages)
    return TokenUsage(prompt, approx_tokens(text))
