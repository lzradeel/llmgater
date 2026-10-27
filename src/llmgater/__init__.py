"""llmgater: one async API in front of OpenAI, Anthropic and OpenAI-compatible servers."""

from llmgater.backends import (
    AnthropicBackend,
    Backend,
    HTTPBackend,
    OpenAIBackend,
    ScriptedBackend,
)
from llmgater.backoff import BackoffPolicy
from llmgater.caching import HashEmbedder, OpenAICompatEmbedder, SimilarityCache, TextEmbedder
from llmgater.core import Gater, PromptInput
from llmgater.exceptions import (
    AllBackendsFailedError,
    BackendError,
    GaterError,
    InjectionBlockedError,
    QuotaExceededError,
    SchemaMismatchError,
)
from llmgater.models import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    StreamDelta,
    TokenUsage,
    approx_tokens,
)
from llmgater.quota import TokenQuota
from llmgater.safety import InjectionScorer, PIIScrubber, SafetyFilter

__version__ = "0.1.0"

__all__ = [
    "AllBackendsFailedError",
    "AnthropicBackend",
    "Backend",
    "BackendError",
    "BackoffPolicy",
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "Gater",
    "GaterError",
    "HTTPBackend",
    "HashEmbedder",
    "InjectionBlockedError",
    "InjectionScorer",
    "OpenAIBackend",
    "OpenAICompatEmbedder",
    "PIIScrubber",
    "PromptInput",
    "QuotaExceededError",
    "SafetyFilter",
    "SchemaMismatchError",
    "ScriptedBackend",
    "SimilarityCache",
    "StreamDelta",
    "TextEmbedder",
    "TokenQuota",
    "TokenUsage",
    "approx_tokens",
]
