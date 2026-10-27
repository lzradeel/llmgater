import math
import time
from collections.abc import Callable, Mapping

from llmgater.exceptions import QuotaExceededError


class TokenQuota:
    """Token caps per account, optionally refilled every `period` seconds."""

    def __init__(
        self,
        caps: Mapping[str, int] | None = None,
        *,
        fallback: int | None = None,
        period: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.fallback = fallback
        self.period = period
        self._caps = dict(caps or {})
        self._clock = clock
        self._consumed: dict[str, int] = {}
        self._refill_at: dict[str, float] = {}

    def cap_for(self, account: str) -> int | None:
        """The account's own cap, else the fallback, else None for unlimited."""
        return self._caps.get(account, self.fallback)

    def set_cap(self, account: str, tokens: int) -> None:
        """Give an account its own cap, replacing any earlier one."""
        self._caps[account] = tokens

    def consumed(self, account: str) -> int:
        """Tokens used in the current period, refilling first if the period is over."""
        if self._clock() >= self._refill_at.get(account, math.inf):
            self.clear(account)
        return self._consumed.get(account, 0)

    def available(self, account: str) -> int | None:
        """Tokens left before the cap, or None when the account is unlimited."""
        cap = self.cap_for(account)
        if cap is None:
            return None
        return max(cap - self.consumed(account), 0)

    def ensure(self, account: str, tokens: int = 0) -> None:
        """Raise QuotaExceededError if `tokens` more would take the account past its cap."""
        cap = self.cap_for(account)
        consumed = self.consumed(account)
        if cap is not None and consumed + tokens > cap:
            raise QuotaExceededError(account, consumed=consumed, requested=tokens, cap=cap)

    def record(self, account: str, tokens: int) -> None:
        """Add tokens to the running total and start the refill clock on first use."""
        self._consumed[account] = self.consumed(account) + tokens
        if self.period is not None and account not in self._refill_at:
            self._refill_at[account] = self._clock() + self.period

    def clear(self, account: str | None = None) -> None:
        """Forget usage for one account, or for every account when none is given."""
        if account is None:
            self._consumed.clear()
            self._refill_at.clear()
            return
        self._consumed.pop(account, None)
        self._refill_at.pop(account, None)
