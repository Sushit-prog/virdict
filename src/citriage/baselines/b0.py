from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from citriage.labels import LabelName
from citriage.parse.logparse import Evidence, ParsedLog

CULPRIT_RE = re.compile(r"(?P<path>[\w.@+-]+(?:[\\/][\w.@+-]+)+)")


@dataclass(frozen=True)
class B0Result:
    label: LabelName
    culprit_file: str | None
    evidence_ids: list[str]


def _culprit_file(evidence: Evidence) -> str | None:
    match = CULPRIT_RE.search(evidence.text)
    if match is None:
        return None
    path = match.group("path").replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    return path or None


def _select(parsed: ParsedLog) -> list[Evidence]:
    candidates = [item for item in parsed.evidence if item.kind != "tail"]
    in_region = [item for item in candidates if item.in_region]
    return in_region if in_region else candidates


def predict(parsed: ParsedLog) -> B0Result:
    items = _select(parsed)
    if not items:
        return B0Result(LabelName.UNKNOWN, None, [])

    rules: tuple[tuple[LabelName, Callable[[Evidence], bool]], ...] = (
        (LabelName.CONFIG_ERROR, lambda item: item.kind == "config"),
        (LabelName.INFRA_FLAKE, lambda item: item.kind == "infra"),
        (LabelName.DEPENDENCY, lambda item: item.kind == "dependency"),
        (
            LabelName.LINT_FORMAT,
            lambda item: (
                item.kind in {"ruff_lint", "format", "lint_format"}
                or (item.kind == "ratchet" and item.tool == "ruff")
            ),
        ),
        (
            LabelName.TYPE_ERROR,
            lambda item: (
                item.kind == "type_error"
                or (item.kind == "ratchet" and item.tool == "basedpyright")
            ),
        ),
        (
            LabelName.TEST_FAILURE,
            lambda item: item.kind.startswith("pytest_") or item.kind == "test_failure",
        ),
    )
    for label, predicate in rules:
        matches = [item for item in items if predicate(item)]
        if matches:
            chosen = next((item for item in matches if _culprit_file(item)), matches[0])
            return B0Result(
                label,
                _culprit_file(chosen),
                [item.id for item in matches],
            )
    return B0Result(LabelName.UNKNOWN, None, [])


__all__ = ["B0Result", "predict"]
