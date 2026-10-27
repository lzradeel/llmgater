import random
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BackoffPolicy:
    """How often a backend is retried and how long to wait between tries."""

    attempts: int = 3
    initial_delay: float = 0.5
    delay_cap: float = 8.0
    randomize: bool = True

    def __post_init__(self) -> None:
        if self.attempts < 1:
            raise ValueError("BackoffPolicy.attempts must be 1 or more")

    def wait_for(self, attempt: int, wait_hint: float | None = None) -> float:
        """Seconds to sleep after the given failed attempt (counted from 1)."""
        if wait_hint is not None:
            return min(wait_hint, self.delay_cap)
        ceiling = min(self.delay_cap, self.initial_delay * 2 ** (attempt - 1))
        # Keep at least half the delay so randomised retries never bunch up near zero.
        return random.uniform(ceiling / 2, ceiling) if self.randomize else ceiling
