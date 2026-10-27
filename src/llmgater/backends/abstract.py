import json
import math
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import httpx

from llmgater.exceptions import BackendError
from llmgater.models import ChatRequest, ChatResponse, StreamDelta

TRANSIENT_HTTP_CODES = (408, 409, 429, 500, 502, 503, 504, 529)
TRANSIENT_ERROR_KINDS = ("overloaded_error", "api_error", "rate_limit_error", "server_error")


class Backend(ABC):
    """One LLM service that the gater can route requests to."""

    name: str
    model: str

    @abstractmethod
    async def generate(self, request: ChatRequest) -> ChatResponse:
        """Return the whole reply in one piece."""

    @abstractmethod
    def generate_stream(self, request: ChatRequest) -> AsyncGenerator[StreamDelta, None]:
        """Yield the reply as it is produced."""

    async def close(self) -> None:
        """Release network resources; does nothing unless overridden."""
        return None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, model={self.model!r})"


class HTTPBackend(Backend):
    """Shared httpx plumbing for backends that speak JSON over HTTP."""

    def __init__(
        self,
        model: str,
        *,
        name: str,
        api_key: str | None,
        base_url: str,
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.model = model
        self.name = name
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.http = client or httpx.AsyncClient()
        self._owns_http = client is None

    def auth_headers(self) -> dict[str, str]:
        """Bearer auth by default; subclasses override this for other schemes."""
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    async def post_json(self, path: str, body: Mapping[str, Any]) -> dict[str, Any]:
        """POST a JSON body and return the decoded JSON object."""
        try:
            reply = await self.http.post(
                self.base_url + path, json=body, headers=self.auth_headers(), timeout=self.timeout
            )
        except httpx.HTTPError as exc:
            raise self.error_from_transport(exc) from exc
        if not reply.is_success:
            raise self.error_from_status(reply)
        return self.parse_body(reply.text)

    async def sse_events(
        self, path: str, body: Mapping[str, Any]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """POST a JSON body and yield every server-sent `data:` event as a dict."""
        try:
            async with self.http.stream(
                "POST",
                self.base_url + path,
                json=body,
                headers=self.auth_headers(),
                timeout=self.timeout,
            ) as reply:
                if not reply.is_success:
                    await reply.aread()
                    raise self.error_from_status(reply)
                async for line in reply.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line.removeprefix("data:").strip()
                    if payload == "[DONE]":
                        return
                    yield self.parse_body(payload)
        except httpx.HTTPError as exc:
            raise self.error_from_transport(exc) from exc

    async def close(self) -> None:
        """Close the HTTP client, but only if this backend created it."""
        if self._owns_http:
            await self.http.aclose()

    def parse_body(self, text: str) -> dict[str, Any]:
        """Decode a JSON object, raising BackendError for junk or an error envelope."""
        try:
            data = json.loads(text)
        except ValueError:
            data = None
        if not isinstance(data, dict):
            raise BackendError(f"unexpected reply format: {text[:100]!r}", backend=self.name)
        problem = data.get("error")
        if isinstance(problem, dict):
            raise BackendError(
                str(problem.get("message") or problem.get("type") or "unknown error"),
                backend=self.name,
                transient=problem.get("type") in TRANSIENT_ERROR_KINDS,
            )
        return data

    @contextmanager
    def guard_parse(self) -> Iterator[None]:
        """Turn KeyError, TypeError and friends from reply parsing into BackendError."""
        try:
            yield
        except (LookupError, TypeError, ValueError, AttributeError) as exc:
            raise BackendError(f"unexpected reply format: {exc!r}", backend=self.name) from exc

    def error_from_status(self, reply: httpx.Response) -> BackendError:
        """Build a BackendError from a non-2xx response."""
        try:
            message = str(reply.json()["error"]["message"])
        except (ValueError, LookupError, TypeError):
            message = f"HTTP {reply.status_code} {reply.reason_phrase}"
        return BackendError(
            message,
            backend=self.name,
            status=reply.status_code,
            transient=reply.status_code in TRANSIENT_HTTP_CODES,
            wait_hint=read_retry_after(reply.headers.get("retry-after")),
        )

    def error_from_transport(self, exc: httpx.HTTPError) -> BackendError:
        """Network failures and timeouts are always treated as transient."""
        if isinstance(exc, httpx.TimeoutException):
            message = "request timed out"
        else:
            message = str(exc) or type(exc).__name__
        return BackendError(message, backend=self.name, transient=True)


def read_retry_after(value: str | None) -> float | None:
    """Parse a numeric Retry-After header; HTTP-date values are ignored."""
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    if not math.isfinite(seconds) or seconds < 0:
        return None
    return seconds
