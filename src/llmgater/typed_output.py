import contextlib
import json
import re
from collections.abc import Awaitable, Callable, Sequence
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from llmgater.exceptions import SchemaMismatchError
from llmgater.models import ChatMessage

ModelT = TypeVar("ModelT", bound=BaseModel)

_CODE_FENCE = re.compile(r"```[\w-]*\s*(.*?)```", re.DOTALL)
_DANGLING_COMMA = re.compile(r",\s*([}\]])")
_CURLY_QUOTES = str.maketrans({"“": '"', "”": '"'})


def inject_schema(messages: Sequence[ChatMessage], schema: type[BaseModel]) -> list[ChatMessage]:
    """Add JSON-only instructions and the schema to the system prompt."""
    compact = json.dumps(schema.model_json_schema(), separators=(",", ":"))
    rules = (
        "Return a single JSON value and nothing else (no prose, no code fences). "
        f"The JSON has to validate against this schema: {compact}"
    )
    if messages and messages[0].role == "system":
        head, *tail = messages
        return [ChatMessage("system", f"{head.content}\n\n{rules}"), *tail]
    return [ChatMessage("system", rules), *messages]


def find_json_span(text: str) -> str:
    """Pull the outermost JSON object or array out of a chatty reply."""
    if fenced := _CODE_FENCE.search(text):
        text = fenced.group(1)
    openers = [pos for pos in (text.find("{"), text.find("[")) if pos >= 0]
    if not openers:
        return text.strip()
    start = min(openers)
    end = text.rfind("}" if text[start] == "{" else "]")
    return text[start : end + 1] if end > start else text[start:].strip()


def coerce_reply(text: str, schema: type[ModelT]) -> ModelT:
    """Validate a reply against the schema, trying once more after cheap local fixes."""
    candidate = find_json_span(text)
    try:
        return schema.model_validate_json(candidate)
    except ValidationError:
        patched = _DANGLING_COMMA.sub(r"\1", candidate.translate(_CURLY_QUOTES))
        if patched != candidate:
            with contextlib.suppress(ValidationError):
                return schema.model_validate_json(patched)
        raise


async def run_extraction(
    ask: Callable[[list[ChatMessage]], Awaitable[str]],
    messages: Sequence[ChatMessage],
    schema: type[ModelT],
    *,
    repair_attempts: int = 2,
) -> ModelT:
    """Ask for JSON and feed validation errors back to the model until it complies."""
    conversation = inject_schema(messages, schema)
    rounds = 0
    while True:
        reply = await ask(conversation)
        try:
            return coerce_reply(reply, schema)
        except ValidationError as exc:
            if rounds >= repair_attempts:
                raise SchemaMismatchError(
                    f"no reply matched {schema.__name__} after {rounds} repair round(s)",
                    last_reply=reply,
                ) from exc
            rounds += 1
            nudge = f"That JSON was rejected:\n{_summarize_errors(exc)}\nSend the fixed JSON only."
            conversation = [
                *conversation,
                ChatMessage("assistant", reply),
                ChatMessage("user", nudge),
            ]


def _summarize_errors(error: ValidationError) -> str:
    """One bullet per validation problem, prefixed with the field path."""
    bullets = []
    for issue in error.errors(include_url=False):
        where = ".".join(str(part) for part in issue["loc"])
        bullets.append(f"* {where}: {issue['msg']}" if where else f"* {issue['msg']}")
    return "\n".join(bullets)
