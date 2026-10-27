from dataclasses import dataclass, field
from typing import Literal

Role = Literal["system", "user", "assistant"]


@dataclass(slots=True)
class ChatMessage:
    """A single turn in a conversation."""

    role: Role
    content: str


@dataclass(slots=True)
class TokenUsage:
    """Token counts reported by a backend for one call."""

    tokens_in: int = 0
    tokens_out: int = 0

    @property
    def total(self) -> int:
        """Prompt and completion tokens combined."""
        return self.tokens_in + self.tokens_out


@dataclass(slots=True)
class ChatRequest:
    """Everything a backend needs to produce a reply."""

    messages: list[ChatMessage]
    temperature: float | None = None
    max_output_tokens: int | None = None


@dataclass(slots=True)
class ChatResponse:
    """A finished reply, tagged with the backend that produced it."""

    text: str
    model: str
    backend: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    stop_reason: str | None = None
    from_cache: bool = False


@dataclass(slots=True)
class StreamDelta:
    """One piece of a streamed reply; the final piece usually carries usage."""

    text: str = ""
    usage: TokenUsage | None = None
    stop_reason: str | None = None


def approx_tokens(text: str) -> int:
    """Rough token count using the four-characters-per-token rule of thumb."""
    return -(-len(text) // 4)
