import pytest
from pydantic import BaseModel, ValidationError

from llmgater.exceptions import SchemaMismatchError
from llmgater.models import ChatMessage
from llmgater.typed_output import coerce_reply, find_json_span, inject_schema, run_extraction


class Book(BaseModel):
    title: str
    pages: int


DUNE = Book(title="Dune", pages=412)


@pytest.mark.parametrize(
    "text",
    [
        '{"title": "Dune", "pages": 412}',
        '```json\n{"title": "Dune", "pages": 412}\n```',
        '```\n{"title": "Dune", "pages": 412}\n```',
        'Of course! Here you go:\n{"title": "Dune", "pages": 412}\nHappy reading.',
        '{"title": "Dune", "pages": 412,\n}',
        "{“title”: “Dune”, “pages”: 412}",
    ],
)
def test_coerce_reply_copes_with_messy_output(text: str) -> None:
    assert coerce_reply(text, Book) == DUNE


def test_coerce_reply_raises_when_unfixable() -> None:
    with pytest.raises(ValidationError):
        coerce_reply('{"title": "Dune", "pages": "plenty"}', Book)


@pytest.mark.parametrize(
    ("text", "span"),
    [
        ('Here: {"shelf": [{"id": 1}, {"id": 2}]} - done.', '{"shelf": [{"id": 1}, {"id": 2}]}'),
        ('List [1, 2] then {"a": 1}', "[1, 2]"),
        ("  no json at all  ", "no json at all"),
        ('cut off {"title": "Du', '{"title": "Du'),
    ],
)
def test_find_json_span(text: str, span: str) -> None:
    assert find_json_span(text) == span


def test_inject_schema_extends_an_existing_system_message() -> None:
    messages = [ChatMessage("system", "Be concise."), ChatMessage("user", "Dune?")]
    prompted = inject_schema(messages, Book)
    assert [message.role for message in prompted] == ["system", "user"]
    assert prompted[0].content.startswith("Be concise.\n\n")
    assert '"pages"' in prompted[0].content
    assert messages[0].content == "Be concise."


def test_inject_schema_adds_a_system_message_when_missing() -> None:
    prompted = inject_schema([ChatMessage("user", "Dune?")], Book)
    assert [message.role for message in prompted] == ["system", "user"]
    assert "JSON" in prompted[0].content


async def test_run_extraction_sends_errors_back_for_repair() -> None:
    replies = iter(['{"title": "Dune"}', '{"title": "Dune", "pages": 412}'])
    seen: list[list[ChatMessage]] = []

    async def ask(messages: list[ChatMessage]) -> str:
        seen.append(messages)
        return next(replies)

    assert await run_extraction(ask, [ChatMessage("user", "Dune?")], Book) == DUNE
    first, second = seen
    assert second[: len(first)] == first
    assert second[-2] == ChatMessage("assistant", '{"title": "Dune"}')
    assert second[-1].role == "user"
    assert "* pages: Field required" in second[-1].content


async def test_run_extraction_gives_up_after_the_repair_budget() -> None:
    replies = iter(["no clue", "still no clue"])
    seen: list[list[ChatMessage]] = []

    async def ask(messages: list[ChatMessage]) -> str:
        seen.append(messages)
        return next(replies)

    with pytest.raises(SchemaMismatchError, match="after 1 repair") as caught:
        await run_extraction(ask, [ChatMessage("user", "Dune?")], Book, repair_attempts=1)
    assert caught.value.last_reply == "still no clue"
    assert len(seen) == 2


async def test_zero_repair_attempts_means_a_single_call() -> None:
    calls = 0

    async def ask(messages: list[ChatMessage]) -> str:
        nonlocal calls
        calls += 1
        return "nope"

    with pytest.raises(SchemaMismatchError):
        await run_extraction(ask, [ChatMessage("user", "Dune?")], Book, repair_attempts=0)
    assert calls == 1
