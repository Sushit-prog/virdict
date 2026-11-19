from __future__ import annotations

import re
from collections.abc import Iterable

REDACTED = "[REDACTED]"

_ADD_MASK = re.compile(r"::add-mask::([^\r\n]*)")
_GITHUB_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_])(?:gh[pousr]_[A-Za-z0-9]{20,255}|github_pat_[A-Za-z0-9_]{20,255})"
    r"(?![A-Za-z0-9_])"
)
_BEARER = re.compile(r"\b(Bearer)\s+[A-Za-z0-9._~+/=-]+", re.IGNORECASE)
_AWS_KEY = re.compile(r"\b(?:AKIA|ASIA|AIDA|AROA|AIPA|ANPA|ANVA|ASCA)[A-Z0-9]{16}\b")
_AWS_SECRET = re.compile(
    r"(aws_secret_access_key\s*[=:]\s*)([\"']?)[A-Za-z0-9/+=]{16,}([\"']?)",
    re.IGNORECASE,
)


def redact(text: str, extra_secrets: Iterable[str] = ()) -> str:
    secrets = tuple(secret for secret in extra_secrets if secret)

    masked_values: list[str] = []

    def replace_mask(match: re.Match[str]) -> str:
        value = match.group(1).strip()
        if value and value not in {REDACTED, "***"}:
            masked_values.append(value)
        return f"::add-mask::{REDACTED}"

    text = _ADD_MASK.sub(replace_mask, text)

    for value in sorted(set(masked_values), key=len, reverse=True):
        text = text.replace(value, REDACTED)
    for value in sorted(set(secrets), key=len, reverse=True):
        text = text.replace(value, REDACTED)

    text = _GITHUB_TOKEN.sub(REDACTED, text)
    text = _BEARER.sub(lambda match: f"{match.group(1)} {REDACTED}", text)
    text = _AWS_KEY.sub(REDACTED, text)
    text = _AWS_SECRET.sub(
        lambda match: f"{match.group(1)}{match.group(2)}{REDACTED}{match.group(3)}",
        text,
    )
    return text


def redact_strings(value: object, extra_secrets: Iterable[str] = ()) -> object:
    secrets = tuple(secret for secret in extra_secrets if secret)
    if isinstance(value, str):
        return redact(value, secrets)
    if isinstance(value, list):
        return [redact_strings(item, secrets) for item in value]
    if isinstance(value, dict):
        return {
            (redact(key, secrets) if isinstance(key, str) else key): redact_strings(item, secrets)
            for key, item in value.items()
        }
    return value
