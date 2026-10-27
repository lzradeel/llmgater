import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import replace

from llmgater.exceptions import InjectionBlockedError
from llmgater.models import ChatMessage

_OCTET = r"(?:25[0-5]|2[0-4]\d|1?\d?\d)"

# Order matters: when two patterns match at the same position, the earlier one wins.
PII_PATTERNS: dict[str, str] = {
    "EMAIL": r"\b[\w.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b",
    "API_KEY": r"\b(?:(?:sk|pk|rk)[-_](?=[\w-]*\d)[\w-]{20,}|AKIA[0-9A-Z]{16})\b",
    "SSN": r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b",
    "CREDIT_CARD": (
        r"\b(?:[2-6]\d{3}(?:[ -]?\d{4}){3}|3[47]\d{2}[ -]?\d{6}[ -]?\d{5}|[2-6]\d{12,18})\b"
    ),
    "PHONE": (
        r"(?<![\w+])(?:(?:\+\d{1,3}|1)[ .-]?)?(?:\(\d{3}\) ?|\d{3}[ .-])\d{3}[ .-]\d{4}\b"
        r"|\+\d{1,3}(?:[ .-]?\d{2,5}){2,4}\b"
    ),
    "IP_ADDRESS": rf"(?<![\w.]){_OCTET}(?:\.{_OCTET}){{3}}(?!\.?\w)",
}

# Each trigger is (weight, pattern); patterns run on lowercased, normalised text.
_TRIGGERS: dict[str, tuple[float, re.Pattern[str]]] = {
    "override_rules": (
        0.8,
        re.compile(
            r"\b(?:ignore|disregard|forget|override)(?: \w+){0,3}? "
            r"(?:all|any|previous|prior|above|earlier|preceding|initial|original|your)"
            r"(?: \w+){0,2}? (?:instructions?|prompts?|directives|guidelines|rules)\b"
        ),
    ),
    "prompt_leak": (
        0.6,
        re.compile(
            r"\b(?:reveal|show|print|repeat|output|display|leak|dump|tell|what is|what's|what are)"
            r"(?: me| us)? (?:your|the) "
            r"(?:system prompt|(?:initial|original|hidden|secret) (?:instructions|prompt))"
        ),
    ),
    "persona_unlock": (
        0.5,
        re.compile(
            r"\b(?:pretend|act as|act like|role[- ]?play|you are now|imagine you are)\b.{0,60}?"
            r"\b(?:unrestricted|unfiltered|uncensored|(?:no|without|free of) (?:any )?"
            r"(?:restrictions|limitations|filters|censorship|guidelines|rules|boundaries))\b"
        ),
    ),
    "jailbreak_mode": (
        0.8,
        re.compile(
            r"\b(?:(?:dan|jailbreak|jailbroken|unrestricted|unfiltered) mode"
            r"|do anything now|you are (?:now )?dan)\b"
        ),
    ),
    "safety_bypass": (
        0.6,
        re.compile(
            r"\b(?:bypass|disable|override|turn off|ignore|circumvent)(?: \w+){0,2}? "
            r"(?:safety|moderation|ethical) "
            r"(?:filters?|guidelines|policies|rules|restrictions|guardrails|protocols)\b"
        ),
    ),
    "template_token": (0.7, re.compile(r"<\|[a-z_]+\|>|\[/?inst\]|<</?sys>>")),
    "instruction_swap": (
        0.35,
        re.compile(r"\b(?:from now on|new instructions|your new (?:task|role|goal) is)\b"),
    ),
}

_INVISIBLE = re.compile(r"[­​-‏⁠-⁤﻿]")


def _luhn_ok(number: str) -> bool:
    """Luhn checksum, used to tell real card numbers apart from other long digit runs."""
    digits = [int(char) for char in reversed(number) if char.isdigit()]
    total = sum(digits[::2])
    for digit in digits[1::2]:
        doubled = digit * 2
        total += doubled - 9 if doubled > 9 else doubled
    return total % 10 == 0


def _canonical(text: str) -> str:
    """NFKC-fold, strip invisible characters, lowercase and collapse whitespace."""
    folded = _INVISIBLE.sub("", unicodedata.normalize("NFKC", text))
    return " ".join(folded.lower().split())


class PIIScrubber:
    """Swaps emails, card numbers, phone numbers and similar PII for [REDACTED:LABEL] tags."""

    def __init__(self, patterns: Mapping[str, str] | None = None) -> None:
        chosen = PII_PATTERNS if patterns is None else patterns
        self.labels = list(chosen)
        self._combined = re.compile(
            "|".join(f"(?P<{label}>{regex})" for label, regex in chosen.items())
        )

    def scrub(self, text: str) -> str:
        """Return the text with every recognised PII span replaced by its tag."""
        if not self.labels:
            return text
        return self._combined.sub(self._tag, text)

    def _tag(self, match: re.Match[str]) -> str:
        """Replacement for one match; digit runs that fail Luhn are left as they are."""
        label = match.lastgroup
        if label == "CREDIT_CARD" and not _luhn_ok(match.group()):
            return match.group()
        return f"[REDACTED:{label}]"


class InjectionScorer:
    """Rates text for prompt-injection phrasing using weighted regex triggers."""

    def __init__(self, cutoff: float = 0.5) -> None:
        self.cutoff = cutoff

    def evaluate(self, text: str) -> tuple[float, list[str]]:
        """Return (score between 0 and 1, names of the triggers that matched)."""
        canonical = _canonical(text)
        matched = [name for name, (_, regex) in _TRIGGERS.items() if regex.search(canonical)]
        # Noisy-OR: every trigger counts as independent evidence, so weak ones add up.
        clean_odds = math.prod(1.0 - _TRIGGERS[name][0] for name in matched)
        return round(1.0 - clean_odds, 3), matched

    def is_suspicious(self, text: str) -> bool:
        """True when the score reaches the cutoff."""
        return self.evaluate(text)[0] >= self.cutoff


class SafetyFilter:
    """Blocks likely prompt injections and scrubs PII before messages reach a backend."""

    def __init__(
        self,
        *,
        scrub_pii: bool = True,
        block_injection: bool = True,
        injection_cutoff: float = 0.5,
        scrubber: PIIScrubber | None = None,
        scorer: InjectionScorer | None = None,
    ) -> None:
        self.scrubber = (scrubber or PIIScrubber()) if scrub_pii else None
        self.scorer = (scorer or InjectionScorer(injection_cutoff)) if block_injection else None

    def screen(self, messages: Sequence[ChatMessage]) -> list[ChatMessage]:
        """Raise on a suspected injection, otherwise return scrubbed copies of the messages."""
        if self.scorer is not None:
            for message in messages:
                if message.role != "user":
                    continue
                score, matched = self.scorer.evaluate(message.content)
                if score >= self.scorer.cutoff:
                    raise InjectionBlockedError(score, matched)
        return [self._scrub(message) for message in messages]

    def _scrub(self, message: ChatMessage) -> ChatMessage:
        """Copy a message, scrubbing it unless it is a trusted system message."""
        if self.scrubber is None or message.role == "system":
            return replace(message)
        return replace(message, content=self.scrubber.scrub(message.content))
