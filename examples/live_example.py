"""Live run with OpenAI first and Anthropic as the standby; both API keys must be set."""

import asyncio

from pydantic import BaseModel

from llmgater import (
    AnthropicBackend,
    Gater,
    OpenAIBackend,
    SafetyFilter,
    SimilarityCache,
    TokenQuota,
)


class SupportTicket(BaseModel):
    area: str
    urgency: int
    summary: str


async def main() -> None:
    gater = Gater(
        [OpenAIBackend("gpt-5-mini"), AnthropicBackend("claude-haiku-5-5")],
        timeout=20,
        cache=SimilarityCache(),
        quota=TokenQuota(fallback=50_000),
        safety=SafetyFilter(),
    )
    async with gater:
        reply = await gater.ask(
            "Explain the circuit breaker pattern in two sentences.", account="orbit"
        )
        print(f"{reply.backend}/{reply.model} ({reply.usage.total} tokens)")
        print(reply.text, end="\n\n")

        async for piece in gater.ask_stream("Write a haiku about retries.", account="orbit"):
            print(piece, end="", flush=True)
        print("\n")

        ticket = await gater.extract(
            "Checkout has returned 502s since this morning's deploy and customers are angry.",
            SupportTicket,
            account="orbit",
        )
        print(ticket.model_dump())


if __name__ == "__main__":
    asyncio.run(main())
