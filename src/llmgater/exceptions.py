class GaterError(Exception):
    """Base class for every error raised by llmgater."""


class BackendError(GaterError):
    """A backend call failed; `transient` says whether a retry might help."""

    def __init__(
        self,
        message: str,
        *,
        backend: str,
        status: int | None = None,
        transient: bool = False,
        wait_hint: float | None = None,
    ) -> None:
        super().__init__(f"[{backend}] {message}")
        self.backend = backend
        self.status = status
        self.transient = transient
        self.wait_hint = wait_hint


class AllBackendsFailedError(GaterError):
    """Every backend was tried and none of them produced a reply."""

    def __init__(self, failures: list[BackendError]) -> None:
        summary = " | ".join(str(failure) for failure in failures)
        super().__init__(f"no backend could serve the request: {summary}")
        self.failures = failures


class QuotaExceededError(GaterError):
    """The request would push an account past its token cap."""

    def __init__(self, account: str, *, consumed: int, requested: int, cap: int) -> None:
        super().__init__(
            f"account {account!r} is out of tokens: "
            f"{consumed} consumed + {requested} requested exceeds the cap of {cap}"
        )
        self.account = account
        self.consumed = consumed
        self.requested = requested
        self.cap = cap


class InjectionBlockedError(GaterError):
    """A user message looked like a prompt-injection attempt and was stopped."""

    def __init__(self, score: float, triggers: list[str]) -> None:
        super().__init__(
            f"blocked as likely prompt injection (score {score:.2f}; {', '.join(triggers)})"
        )
        self.score = score
        self.triggers = triggers


class SchemaMismatchError(GaterError):
    """The model never produced output that matched the requested schema."""

    def __init__(self, message: str, last_reply: str) -> None:
        super().__init__(message)
        self.last_reply = last_reply
