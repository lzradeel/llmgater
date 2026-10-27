import hashlib
import json
import math
import os
import re
import time
import zlib
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import Protocol

import httpx

from llmgater.exceptions import BackendError
from llmgater.models import ChatMessage, ChatResponse

_TOKEN_RE = re.compile(r"\w+")
_VECTOR_MEMO_SIZE = 64
_OPENAI_BASE = "https://api.openai.com/v1"


class TextEmbedder(Protocol):
    """Anything that can turn a piece of text into a vector."""

    async def embed(self, text: str) -> list[float]: ...


class HashEmbedder:
    """Offline embedder that hashes words and word pairs into a fixed-size vector."""

    def __init__(self, size: int = 256) -> None:
        self.size = size

    async def embed(self, text: str) -> list[float]:
        """Signed feature hashing over unigrams and bigrams, scaled to unit length."""
        tokens = _TOKEN_RE.findall(text.lower())
        features = tokens + [f"{left} {right}" for left, right in pairwise(tokens)]
        vector = [0.0] * self.size
        for feature in features:
            bucket = zlib.crc32(feature.encode())
            # The top bit picks the sign so that unrelated collisions tend to cancel out.
            vector[bucket % self.size] += 1.0 if bucket >> 31 else -1.0
        return _unit_vector(vector)


class OpenAICompatEmbedder:
    """Embeddings from OpenAI or any server that speaks its /embeddings API."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if base_url is None:
            # The env key is only used when talking to OpenAI itself.
            api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.base_url = (base_url or _OPENAI_BASE).rstrip("/")
        self._auth = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = client or httpx.AsyncClient(timeout=30.0)
        self._owns_http = client is None

    async def embed(self, text: str) -> list[float]:
        """Fetch one embedding, mapping HTTP and shape problems to BackendError."""
        try:
            reply = await self._http.post(
                f"{self.base_url}/embeddings",
                json={"model": self.model, "input": text},
                headers=self._auth,
            )
        except httpx.HTTPError as exc:
            raise BackendError(
                f"embedding call failed: {exc!r}", backend="openai", transient=True
            ) from exc
        if reply.is_error:
            code = reply.status_code
            raise BackendError(
                f"embedding call returned HTTP {code}",
                backend="openai",
                status=code,
                transient=code in (408, 409, 429) or code >= 500,
            )
        try:
            return [float(value) for value in reply.json()["data"][0]["embedding"]]
        except (ValueError, LookupError, TypeError) as exc:
            raise BackendError("embedding reply had an unexpected shape", backend="openai") from exc

    async def close(self) -> None:
        """Close the HTTP client, but only if this embedder created it."""
        if self._owns_http:
            await self._http.aclose()


@dataclass(slots=True)
class _Slot:
    vector: list[float]
    response: ChatResponse
    expires_at: float


class SimilarityCache:
    """In-memory LRU cache that also matches prompts by embedding similarity."""

    def __init__(
        self,
        embedder: TextEmbedder | None = None,
        *,
        min_similarity: float = 0.95,
        capacity: int = 1024,
        ttl: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.embedder: TextEmbedder = embedder or HashEmbedder()
        self.min_similarity = min_similarity
        self.capacity = capacity
        self.ttl = ttl
        self.hit_count = 0
        self.miss_count = 0
        self._clock = clock
        self._slots: OrderedDict[tuple[str, str], _Slot] = OrderedDict()
        self._vector_memo: OrderedDict[str, list[float]] = OrderedDict()

    def __len__(self) -> int:
        return len(self._slots)

    async def lookup(self, prompt: str, scope: str = "") -> ChatResponse | None:
        """Return a copy of the cached reply for this prompt (exact or similar), or None."""
        self._evict_expired()
        key: tuple[str, str] | None = (scope, prompt)
        if key not in self._slots:
            key = self._nearest(await self._vector(prompt), scope)
        if key is None:
            self.miss_count += 1
            return None
        self.hit_count += 1
        self._slots.move_to_end(key)
        return replace(self._slots[key].response, from_cache=True)

    async def store(self, prompt: str, response: ChatResponse, scope: str = "") -> None:
        """Save a reply, evicting the least recently used entries beyond capacity."""
        vector = await self._vector(prompt)
        self._evict_expired()
        key = (scope, prompt)
        expires_at = math.inf if self.ttl is None else self._clock() + self.ttl
        self._slots[key] = _Slot(vector, response, expires_at)
        self._slots.move_to_end(key)
        while len(self._slots) > self.capacity:
            self._slots.popitem(last=False)

    def flush(self) -> None:
        """Drop every entry and reset the hit and miss counters."""
        self._slots.clear()
        self._vector_memo.clear()
        self.hit_count = 0
        self.miss_count = 0

    async def _vector(self, prompt: str) -> list[float]:
        """Embed a prompt, reusing the vector if the same prompt was embedded recently."""
        if (memo := self._vector_memo.get(prompt)) is not None:
            return memo
        vector = _unit_vector(await self.embedder.embed(prompt))
        self._vector_memo[prompt] = vector
        if len(self._vector_memo) > _VECTOR_MEMO_SIZE:
            self._vector_memo.popitem(last=False)
        return vector

    def _nearest(self, vector: list[float], scope: str) -> tuple[str, str] | None:
        """Most similar entry in the scope that clears min_similarity, if there is one."""
        winner: tuple[str, str] | None = None
        best = self.min_similarity
        for key, slot in self._slots.items():
            if key[0] != scope:
                continue
            similarity = _dot(vector, slot.vector)
            if similarity >= best:
                winner, best = key, similarity
        return winner

    def _evict_expired(self) -> None:
        """Remove entries whose TTL has run out."""
        now = self._clock()
        stale = [key for key, slot in self._slots.items() if slot.expires_at <= now]
        for key in stale:
            del self._slots[key]


def split_cache_key(messages: Sequence[ChatMessage]) -> tuple[str, str]:
    """Split a conversation into (hash of the earlier turns, text of the final turn)."""
    *history, final = messages
    encoded = json.dumps([[message.role, message.content] for message in history])
    return hashlib.sha256(encoded.encode()).hexdigest()[:16], final.content


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    """Dot product, which equals cosine similarity for unit vectors."""
    return sum(a * b for a, b in zip(left, right, strict=True))


def _unit_vector(vector: list[float]) -> list[float]:
    """Scale a vector to length 1, leaving the zero vector untouched."""
    length = math.hypot(*vector)
    return [value / length for value in vector] if length else vector
