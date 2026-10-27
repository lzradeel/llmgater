import asyncio
import logging
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Mapping, Sequence
from contextlib import aclosing
from typing import Self, TypeVar

from pydantic import BaseModel

from llmgater.backends import Backend
from llmgater.backoff import BackoffPolicy
from llmgater.caching import SimilarityCache, split_cache_key
from llmgater.exceptions import AllBackendsFailedError, BackendError
from llmgater.models import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    StreamDelta,
    TokenUsage,
    approx_tokens,
)
from llmgater.quota import TokenQuota
from llmgater.safety import SafetyFilter
from llmgater.typed_output import run_extraction

ModelT = TypeVar("ModelT", bound=BaseModel)
ResultT = TypeVar("ResultT")

PromptInput = str | Sequence[ChatMessage | Mapping[str, str]]

log = logging.getLogger("llmgater")


class Gater:
    """One async front door for several LLM backends, with failover, caching, quotas and safety."""

    def __init__(
        self,
        backends: Sequence[Backend],
        *,
        backoff: BackoffPolicy | None = None,
        timeout: float | None = 30.0,
        cache: SimilarityCache | None = None,
        quota: TokenQuota | None = None,
        safety: SafetyFilter | None = None,
    ) -> None:
        if not backends:
            raise ValueError("Gater requires at least one backend")
        self.backends = list(backends)
        self.backoff = backoff or BackoffPolicy()
        self.timeout = timeout
        self.cache = cache
        self.quota = quota
        self.safety = safety

    async def ask(
        self,
        prompt: PromptInput,
        *,
        account: str = "default",
        instructions: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        allow_cache: bool = True,
    ) -> ChatResponse:
        """Send a prompt and return the full reply, failing over between backends."""
        messages = self._build_messages(prompt, instructions)
        if allow_cache and (hit := await self._cache_read(account, messages)) is not None:
            return hit
        self._enforce_quota(account, messages)
        request = ChatRequest(messages, temperature, max_output_tokens)
        response = await self._generate_with_failover(request)
        self._bill(account, response.usage)
        if allow_cache:
            await self._cache_write(account, messages, response)
        return response

    async def ask_stream(
        self,
        prompt: PromptInput,
        *,
        account: str = "default",
        instructions: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        allow_cache: bool = True,
    ) -> AsyncGenerator[str, None]:
        """Yield reply text as it arrives; usage is billed even if the caller stops early."""
        messages = self._build_messages(prompt, instructions)
        if allow_cache and (hit := await self._cache_read(account, messages)) is not None:
            yield hit.text
            return
        self._enforce_quota(account, messages)
        request = ChatRequest(messages, temperature, max_output_tokens)
        source: Backend | None = None
        deltas: list[StreamDelta] = []
        response: ChatResponse | None = None
        try:
            async with aclosing(self._stream_with_failover(request)) as feed:
                async for backend, delta in feed:
                    source = backend
                    deltas.append(delta)
                    if delta.text:
                        yield delta.text
        finally:
            if source is not None:
                response = merge_deltas(source, request, deltas)
                self._bill(account, response.usage)
        if allow_cache and response is not None:
            await self._cache_write(account, messages, response)

    async def extract(
        self,
        prompt: PromptInput,
        schema: type[ModelT],
        *,
        account: str = "default",
        instructions: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        repair_attempts: int = 2,
    ) -> ModelT:
        """Return a validated Pydantic model, asking the model to repair invalid JSON."""
        rounds = 0

        async def ask_once(conversation: list[ChatMessage]) -> str:
            # Only the first round may come from the cache; repairs need a fresh answer.
            nonlocal rounds
            rounds += 1
            response = await self.ask(
                conversation,
                account=account,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                allow_cache=rounds == 1,
            )
            return response.text

        messages = normalize_prompt(prompt, instructions)
        return await run_extraction(ask_once, messages, schema, repair_attempts=repair_attempts)

    async def close(self) -> None:
        """Close every backend's network resources."""
        for backend in self.backends:
            await backend.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    def _build_messages(self, prompt: PromptInput, instructions: str | None) -> list[ChatMessage]:
        """Normalise the prompt and pass it through the safety filter, if one is set."""
        messages = normalize_prompt(prompt, instructions)
        if self.safety is None:
            return messages
        return self.safety.screen(messages)

    async def _generate_with_failover(self, request: ChatRequest) -> ChatResponse:
        """Try each backend in turn, retrying transient errors before moving on."""
        failures: list[BackendError] = []
        for backend in self.backends:
            self._log_handover(failures, backend)
            for attempt in range(1, self.backoff.attempts + 1):
                try:
                    return await self._with_deadline(backend, backend.generate(request))
                except BackendError as exc:
                    failures.append(exc)
                    if not await self._pause_before_retry(attempt, exc):
                        break
        raise AllBackendsFailedError(failures)

    async def _stream_with_failover(
        self, request: ChatRequest
    ) -> AsyncGenerator[tuple[Backend, StreamDelta], None]:
        """Like _generate_with_failover, but stops switching once any text has been sent."""
        failures: list[BackendError] = []
        for backend in self.backends:
            self._log_handover(failures, backend)
            for attempt in range(1, self.backoff.attempts + 1):
                emitted_text = False
                try:
                    async with aclosing(backend.generate_stream(request)) as deltas:
                        while (delta := await self._pull(backend, deltas)) is not None:
                            emitted_text = emitted_text or bool(delta.text)
                            yield backend, delta
                    return
                except BackendError as exc:
                    if emitted_text:
                        raise
                    failures.append(exc)
                    if not await self._pause_before_retry(attempt, exc):
                        break
        raise AllBackendsFailedError(failures)

    def _log_handover(self, failures: list[BackendError], backend: Backend) -> None:
        """Log the switch when an earlier backend has already failed."""
        if failures:
            log.warning("%s; switching to backend %s", failures[-1], backend.name)

    async def _pull(
        self, backend: Backend, deltas: AsyncIterator[StreamDelta]
    ) -> StreamDelta | None:
        """Next delta from a stream (None at the end), under the per-chunk deadline."""
        return await self._with_deadline(backend, anext(deltas, None))

    async def _with_deadline(self, backend: Backend, step: Awaitable[ResultT]) -> ResultT:
        """Await one step, turning a timeout into a transient BackendError."""
        try:
            async with asyncio.timeout(self.timeout):
                return await step
        except TimeoutError:
            raise BackendError(
                f"timed out after {self.timeout}s", backend=backend.name, transient=True
            ) from None

    async def _pause_before_retry(self, attempt: int, failure: BackendError) -> bool:
        """Sleep before the next attempt; returns False when this backend should be dropped."""
        if not failure.transient or attempt >= self.backoff.attempts:
            return False
        delay = self.backoff.wait_for(attempt, failure.wait_hint)
        log.warning("%s; retry %d in %.2fs", failure, attempt + 1, delay)
        await asyncio.sleep(delay)
        return True

    async def _cache_read(self, account: str, messages: list[ChatMessage]) -> ChatResponse | None:
        """Look the conversation up in the cache; a failing cache counts as a miss."""
        if self.cache is None:
            return None
        history_hash, prompt = split_cache_key(messages)
        try:
            return await self.cache.lookup(prompt, f"{account}:{history_hash}")
        except BackendError as exc:
            log.warning("cache read skipped: %s", exc)
            return None

    async def _cache_write(
        self, account: str, messages: list[ChatMessage], response: ChatResponse
    ) -> None:
        """Store a reply; a failing embedder never fails the request."""
        if self.cache is None:
            return
        history_hash, prompt = split_cache_key(messages)
        try:
            await self.cache.store(prompt, response, f"{account}:{history_hash}")
        except BackendError as exc:
            log.warning("cache write skipped: %s", exc)

    def _enforce_quota(self, account: str, messages: list[ChatMessage]) -> None:
        """Refuse the request up front if the prompt alone would exceed the quota."""
        if self.quota is not None:
            self.quota.ensure(account, count_prompt_tokens(messages))

    def _bill(self, account: str, usage: TokenUsage) -> None:
        """Charge the account for the tokens that were actually used."""
        if self.quota is not None:
            self.quota.record(account, usage.total)


def normalize_prompt(prompt: PromptInput, instructions: str | None = None) -> list[ChatMessage]:
    """Accept a string or a list of messages/dicts and return ChatMessage objects."""
    if isinstance(prompt, str):
        messages = [ChatMessage("user", prompt)]
    else:
        messages = [coerce_message(item) for item in prompt]
    if not messages:
        raise ValueError("prompt is empty: at least one message is required")
    if instructions is None:
        return messages
    return [ChatMessage("system", instructions), *messages]


def coerce_message(item: ChatMessage | Mapping[str, str]) -> ChatMessage:
    """Turn a {"role": ..., "content": ...} mapping into a ChatMessage."""
    match item:
        case ChatMessage():
            return item
        case {"role": "system" | "user" | "assistant" as role, "content": str(content)}:
            return ChatMessage(role, content)
        case _:
            raise ValueError(f"unsupported message: {item!r}")


def count_prompt_tokens(messages: Sequence[ChatMessage]) -> int:
    """Estimated prompt size, used for the up-front quota check."""
    return sum(approx_tokens(message.content) for message in messages)


def merge_deltas(
    backend: Backend, request: ChatRequest, deltas: Sequence[StreamDelta]
) -> ChatResponse:
    """Fold streamed deltas into one ChatResponse, estimating usage if none was reported."""
    text = "".join(delta.text for delta in deltas)
    usage = TokenUsage(count_prompt_tokens(request.messages), approx_tokens(text))
    stop_reason: str | None = None
    for delta in deltas:
        usage = delta.usage or usage
        stop_reason = delta.stop_reason or stop_reason
    return ChatResponse(text, backend.model, backend.name, usage, stop_reason)
