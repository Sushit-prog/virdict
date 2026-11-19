from __future__ import annotations

import hashlib
import re

from citriage.parse.logparse import ParsedLog

TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?")
PATH_RE = re.compile(r"(?:[A-Za-z0-9_.+-]+[\\/])+[A-Za-z0-9_.+-]+")
HEX_RE = re.compile(r"\b(?:0x)?[0-9A-Fa-f]{7,}\b")
DIGIT_RE = re.compile(r"\d+")
WHITESPACE_RE = re.compile(r"\s+")


def normalize_for_signature(text: str) -> str:
    normalized = TIMESTAMP_RE.sub("<time>", text)
    normalized = PATH_RE.sub("<path>", normalized)
    normalized = HEX_RE.sub("<hex>", normalized)
    normalized = DIGIT_RE.sub("#", normalized)
    return WHITESPACE_RE.sub(" ", normalized).strip().lower()


def signature(tool: str, text: str) -> str:
    payload = f"{tool}|{normalize_for_signature(text)}"
    return hashlib.sha1(payload.encode()).hexdigest()


def signature_from_log(parsed: ParsedLog) -> str:
    if not parsed.evidence:
        return signature("", "")
    first = parsed.evidence[0]
    return signature(first.tool, first.text)


__all__ = ["normalize_for_signature", "signature", "signature_from_log"]
