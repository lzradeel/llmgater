import json
import math
from unittest.mock import Mock

import httpx
import pytest

from llmgater.caching import HashEmbedder, OpenAICompatEmbedder, SimilarityCache, split_cache_key
from llmgater.exceptions import BackendError
from llmgater.models import ChatMessage, ChatResponse

EVEREST = "How tall is Mount Everest?"


def answer(text: str = "8,849 m") -> ChatResponse:
    return ChatResponse(text=text, model="scripted-1", backend="scripted")


class TallyEmbedder(HashEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def embed(self, text: str) -> list[float]:
        self.calls += 1
        return await super().embed(text)


async def test_exact_match_returns_a_flagged_copy() -> None:
    cache = SimilarityCache()
    stored = answer()
    assert await cache.lookup(EVEREST) is None
    await cache.store(EVEREST, stored)

    hit = await cache.lookup(EVEREST)
    assert hit is not None
    assert hit is not stored
    assert hit.from_cache and not stored.from_cache
    assert hit.text == "8,849 m"
    assert (cache.hit_count, cache.miss_count) == (1, 1)


@pytest.mark.parametrize(
    ("stored", "asked", "hit"),
    [
        (EVEREST, "how tall is mount everest", True),
        (
            "Can you explain how a binary search tree works?",
            "can you explain how a binary search tree works, please?",
            True,
        ),
        (EVEREST, "How tall is Mount Fuji?", False),
        ("Convert 100 miles to kilometres", "Convert 100 kilometres to miles", False),
        ("What year did the Berlin Wall fall?", "What year did the Berlin Wall go up?", False),
    ],
)
async def test_similarity_matching(stored: str, asked: str, hit: bool) -> None:
    cache = SimilarityCache()
    await cache.store(stored, answer())
    assert (await cache.lookup(asked) is not None) is hit


@pytest.mark.parametrize(("min_similarity", "hit"), [(0.95, False), (0.9, True)])
async def test_min_similarity_controls_how_loose_a_match_can_be(
    min_similarity: float, hit: bool
) -> None:
    cache = SimilarityCache(min_similarity=min_similarity)
    await cache.store("What does the GIL do in CPython?", answer("It serialises bytecode."))
    found = await cache.lookup("what does the gil do in cpython, exactly?")
    assert (found is not None) is hit


async def test_scopes_never_see_each_other() -> None:
    cache = SimilarityCache()
    await cache.store(EVEREST, answer(), scope="orbit")
    assert await cache.lookup(EVEREST, scope="lumen") is None
    assert await cache.lookup(EVEREST, scope="orbit") is not None


async def test_entries_disappear_once_the_ttl_runs_out() -> None:
    clock = Mock(return_value=1000.0)
    cache = SimilarityCache(ttl=30, clock=clock)
    await cache.store(EVEREST, answer())
    clock.return_value = 1029.5
    assert await cache.lookup(EVEREST) is not None
    clock.return_value = 1030.0
    assert await cache.lookup(EVEREST) is None
    assert len(cache) == 0


async def test_least_recently_used_entry_is_evicted() -> None:
    cache = SimilarityCache(capacity=3)
    for word in ("alpha", "bravo", "charlie"):
        await cache.store(f"{word} question", answer(word))
    assert await cache.lookup("alpha question") is not None
    await cache.store("delta question", answer("delta"))

    assert len(cache) == 3
    assert await cache.lookup("bravo question") is None
    for word in ("alpha", "charlie", "delta"):
        assert await cache.lookup(f"{word} question") is not None


async def test_a_prompt_is_embedded_only_once_per_miss_and_store() -> None:
    embedder = TallyEmbedder()
    cache = SimilarityCache(embedder)
    assert await cache.lookup(EVEREST) is None
    await cache.store(EVEREST, answer())
    assert await cache.lookup(EVEREST) is not None
    assert embedder.calls == 1


async def test_flush_empties_the_cache_and_resets_counters() -> None:
    cache = SimilarityCache()
    await cache.store(EVEREST, answer())
    await cache.lookup(EVEREST)
    await cache.lookup("unrelated")
    cache.flush()
    assert len(cache) == 0
    assert (cache.hit_count, cache.miss_count) == (0, 0)
    assert await cache.lookup(EVEREST) is None


async def test_hash_embedder_is_unit_length_and_ignores_case_and_punctuation() -> None:
    embedder = HashEmbedder(size=32)
    vector = await embedder.embed("Pack my box with five dozen jugs")
    assert len(vector) == 32
    assert math.isclose(sum(value * value for value in vector), 1.0)
    assert vector == await HashEmbedder(size=32).embed("pack MY box, with five dozen jugs?!")
    assert await embedder.embed("...") == [0.0] * 32


def test_split_cache_key_separates_history_from_the_last_turn() -> None:
    rules = ChatMessage("system", "Reply in metres.")
    question = ChatMessage("user", EVEREST)
    history_hash, prompt = split_cache_key([rules, question])

    assert prompt == EVEREST
    assert split_cache_key([rules, ChatMessage("user", "And K2?")])[0] == history_hash
    assert split_cache_key([ChatMessage("system", "Reply in feet."), question])[0] != history_hash
    assert split_cache_key([question])[0] != history_hash
    assert len(history_hash) == 16


async def test_openai_compat_embedder_sends_the_expected_request() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"data": [{"embedding": [0.28, 0.96]}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    embedder = OpenAICompatEmbedder("text-embedding-3-large", api_key="sk-unit", client=client)

    assert await embedder.embed("lighthouse") == [0.28, 0.96]
    assert str(seen[0].url) == "https://api.openai.com/v1/embeddings"
    assert seen[0].headers["authorization"] == "Bearer sk-unit"
    assert json.loads(seen[0].content) == {"model": "text-embedding-3-large", "input": "lighthouse"}


async def test_env_key_is_only_sent_to_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"data": [{"embedding": [1.0]}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    await OpenAICompatEmbedder(client=client).embed("hi")
    await OpenAICompatEmbedder(base_url="http://localhost:8000/v1/", client=client).embed("hi")

    assert seen[0].headers["authorization"] == "Bearer sk-from-env"
    assert str(seen[1].url) == "http://localhost:8000/v1/embeddings"
    assert "authorization" not in seen[1].headers


@pytest.mark.parametrize(
    ("reply", "transient", "status"),
    [
        (httpx.Response(502), True, 502),
        (httpx.Response(429), True, 429),
        (httpx.Response(403, json={"error": {"message": "forbidden"}}), False, 403),
        (httpx.Response(200, json={"data": [{"vector": [1.0]}]}), False, None),
    ],
)
async def test_embedder_failures_become_backend_errors(
    reply: httpx.Response, transient: bool, status: int | None
) -> None:
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: reply))
    with pytest.raises(BackendError) as caught:
        await OpenAICompatEmbedder(api_key="sk-unit", client=client).embed("hi")
    assert caught.value.transient is transient
    assert caught.value.status == status


async def test_embedder_network_error_is_transient() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with pytest.raises(BackendError) as caught:
        await OpenAICompatEmbedder(api_key="sk-unit", client=client).embed("hi")
    assert caught.value.transient
