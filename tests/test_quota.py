from unittest.mock import Mock

import pytest

from llmgater.exceptions import QuotaExceededError
from llmgater.quota import TokenQuota


def test_no_caps_means_unlimited() -> None:
    quota = TokenQuota()
    quota.record("orbit", 5_000_000_000)
    quota.ensure("orbit", 5_000_000_000)
    assert quota.cap_for("orbit") is None
    assert quota.available("orbit") is None


def test_account_cap_takes_priority_over_fallback() -> None:
    quota = TokenQuota({"orbit": 250}, fallback=25)
    assert quota.cap_for("orbit") == 250
    assert quota.cap_for("lumen") == 25
    quota.set_cap("lumen", 75)
    assert quota.cap_for("lumen") == 75


def test_record_accumulates_and_available_bottoms_out_at_zero() -> None:
    quota = TokenQuota(fallback=200)
    for spend in (40, 60, 25):
        quota.record("orbit", spend)
    assert quota.consumed("orbit") == 125
    assert quota.available("orbit") == 75
    assert quota.consumed("lumen") == 0
    quota.record("orbit", 500)
    assert quota.available("orbit") == 0


@pytest.mark.parametrize(
    ("spent", "asking", "allowed"),
    [(150, 50, True), (150, 51, False), (0, 201, False), (200, 0, True)],
)
def test_ensure_respects_the_exact_boundary(spent: int, asking: int, allowed: bool) -> None:
    quota = TokenQuota({"orbit": 200})
    quota.record("orbit", spent)
    if allowed:
        quota.ensure("orbit", asking)
    else:
        with pytest.raises(QuotaExceededError):
            quota.ensure("orbit", asking)


def test_error_reports_every_number() -> None:
    quota = TokenQuota({"orbit": 200})
    quota.record("orbit", 180)
    with pytest.raises(QuotaExceededError) as caught:
        quota.ensure("orbit", 30)
    err = caught.value
    assert (err.account, err.consumed, err.requested, err.cap) == ("orbit", 180, 30, 200)
    assert "'orbit'" in str(err)


def test_period_is_measured_from_the_first_charge() -> None:
    clock = Mock(return_value=100.0)
    quota = TokenQuota(fallback=50, period=10, clock=clock)
    quota.record("orbit", 30)
    clock.return_value = 105.0
    quota.record("orbit", 20)
    clock.return_value = 109.9
    with pytest.raises(QuotaExceededError):
        quota.ensure("orbit", 1)
    clock.return_value = 110.0
    assert quota.consumed("orbit") == 0
    quota.ensure("orbit", 50)


def test_next_period_starts_at_the_next_charge_after_a_refill() -> None:
    clock = Mock(return_value=0.0)
    quota = TokenQuota(fallback=50, period=10, clock=clock)
    quota.record("orbit", 50)
    clock.return_value = 12.0
    quota.record("orbit", 5)
    clock.return_value = 21.0
    assert quota.consumed("orbit") == 5
    clock.return_value = 22.0
    assert quota.consumed("orbit") == 0


def test_clear_one_account_or_everyone() -> None:
    quota = TokenQuota()
    quota.record("orbit", 7)
    quota.record("lumen", 11)
    quota.clear("orbit")
    assert (quota.consumed("orbit"), quota.consumed("lumen")) == (0, 11)
    quota.clear()
    assert quota.consumed("lumen") == 0
