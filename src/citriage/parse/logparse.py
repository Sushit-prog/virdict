from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\s*")
EXIT_CODE_RE = re.compile(r"^##\[error\]Process completed with exit code \d+\.?$")
GROUP_RUN_PREFIX = "##[group]Run "
CLEANUP_PREFIXES = (
    "##[group]Post job cleanup",
    "Post job cleanup",
    "##[group]Post ",
    "##[finish-job]",
    "Cleaning up workspace",
    "Post Run",
)

PYTEST_SUMMARY_RE = re.compile(r"^(FAILED|ERROR)\s+(\S+::\S+)\s+-\s+(.*)$")
PYTEST_COUNT_RE = re.compile(r"=+\s*\d+\s+failed")
PYTEST_ANNOTATION_RE = re.compile(r"^##\[error\]\s*\S+::\S+")
RUFF_RE = re.compile(r"^(.+?):(\d+):(\d+): ([A-Z]+\d+) (.+)$")
MYPY_RE = re.compile(r"^(.+?):(\d+):(?:\d+:)? error: (.*)$")
PYRIGHT_RE = re.compile(r"^(.+?):(\d+):(\d+) - error: (.*)$")
FORMAT_RE = re.compile(r"would reformat (.+)$", re.IGNORECASE)

DEPENDENCY_PATTERNS = (
    re.compile(r"^ERROR:"),
    re.compile(r"^error: No solution found"),
    re.compile(r"Could not find a version"),
    re.compile(r"SolverProblemError"),
    re.compile(r"Failed building wheel"),
)
INFRA_PATTERNS = (
    re.compile(r"\b(?:timed out|timeout waiting|TimeoutError)\b", re.IGNORECASE),
    re.compile(r"\b(?:ConnectionError|ReadTimeout)\b"),
    re.compile(r"The operation was canceled", re.IGNORECASE),
    re.compile(r"runner has received a shutdown signal", re.IGNORECASE),
    re.compile(r"\b429\b.*(?:rate limit|too many requests)", re.IGNORECASE),
    re.compile(r"(?:rate limit|too many requests).*\b429\b", re.IGNORECASE),
    re.compile(
        r"\b(?:502|503|504)\b.*(?:bad gateway|service unavailable|gateway timeout)", re.IGNORECASE
    ),
    re.compile(
        r"(?:bad gateway|service unavailable|gateway timeout).*\b(?:502|503|504)\b", re.IGNORECASE
    ),
    re.compile(r"No space left on device", re.IGNORECASE),
    re.compile(r"Installation of .* failed", re.IGNORECASE),
    re.compile(r"IP allow ?list", re.IGNORECASE),
)
CONFIG_PATTERNS = (
    "unable to resolve action",
    "resource not accessible by integration",
    "invalid workflow file",
    "input required and not supplied",
    "permission denied",
    "permission error",
    "secret not found",
)
TOOL_PATTERNS = {
    "pytest": re.compile(r"\bpytest\b|py\.test"),
    "ruff": re.compile(r"\bruff\b"),
    "mypy": re.compile(r"\bmypy\b"),
    "pyright": re.compile(r"\bpyright\b"),
    "uv": re.compile(r"\buv\b"),
    "poetry": re.compile(r"\bpoetry\b"),
    "pip": re.compile(r"\bpip\b"),
}

KIND_PRIORITY = {
    "config": 0,
    "infra": 1,
    "dependency": 2,
    "format": 3,
    "ruff_lint": 4,
    "type_error": 5,
    "pytest_summary": 6,
    "pytest_count": 7,
    "pytest_assertion": 8,
    "pytest_annotation": 9,
    "traceback": 10,
    "error_annotation": 11,
}

MAX_EVIDENCE = 40
MAX_TEXT = 300


@dataclass(frozen=True)
class Evidence:
    id: str
    kind: str
    line_no: int
    text: str
    tool: str
    in_region: bool


@dataclass(frozen=True)
class ParsedLog:
    region_start: int
    region_end: int
    evidence: list[Evidence]
    tools_detected: list[str]
    has_nontrivial_error: bool


def _normalize(line: str) -> str:
    return TIMESTAMP_RE.sub("", ANSI_RE.sub("", line))


def _cleanup_index(norms: list[str]) -> int:
    for index, line in enumerate(norms):
        stripped = line.strip()
        if any(stripped.startswith(prefix) for prefix in CLEANUP_PREFIXES):
            return index
    return len(norms)


def _find_region(norms: list[str]) -> tuple[int, int, bool]:
    if not norms:
        return 1, 0, True
    limit = _cleanup_index(norms)
    anchor: int | None = None
    for index in range(limit):
        if EXIT_CODE_RE.match(norms[index].strip()):
            anchor = index
    if anchor is None:
        start = max(0, limit - 300)
        end = limit - 1
        return start + 1, max(start + 1, end + 1), False
    start = anchor
    for index in range(anchor - 1, -1, -1):
        if norms[index].strip().startswith(GROUP_RUN_PREFIX):
            start = index
            break
    return start + 1, anchor + 1, True


def _nontrivial_error_lines(norms: list[str]) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for number, line in enumerate(norms, start=1):
        if "##[error]" not in line:
            continue
        message = line.split("##[error]", 1)[1].strip()
        if not message or message.startswith("Process completed with exit code"):
            continue
        found.append((number, line.strip()))
    return found


def _detect_tools(region_norms: list[str]) -> list[str]:
    return sorted(
        name
        for name, pattern in TOOL_PATTERNS.items()
        if any(pattern.search(line) for line in region_norms)
    )


def _dependency_tool(tools: list[str]) -> str:
    for name in ("uv", "poetry", "pip"):
        if name in tools:
            return name
    return "pip"


def _traceback_line(norms: list[str]) -> tuple[int, str] | None:
    start: int | None = None
    for index, line in enumerate(norms):
        if "Traceback (most recent call last):" in line:
            start = index
    if start is None:
        return None
    last = start
    for index in range(start + 1, len(norms)):
        if not norms[index].strip():
            break
        last = index
    return last + 1, norms[last].strip()


def parse_log(text: str) -> ParsedLog:
    lines = text.splitlines()
    norms = [_normalize(line) for line in lines]
    region_start, region_end, _ = _find_region(norms)
    region_norms = norms[region_start - 1 : region_end]
    tools = _detect_tools(region_norms)
    dep_tool = _dependency_tool(tools)

    candidates: list[tuple[str, str, int, str]] = []

    def add(kind: str, tool: str, number: int, line: str) -> None:
        candidates.append((kind, tool, number, line.strip()[:MAX_TEXT]))

    has_pytest_summary = False
    for number, line in enumerate(norms, start=1):
        if PYTEST_SUMMARY_RE.match(line):
            has_pytest_summary = True
            add("pytest_summary", "pytest", number, line)
        elif PYTEST_COUNT_RE.search(line) and "failed" in line:
            add("pytest_count", "pytest", number, line)
        elif PYTEST_ANNOTATION_RE.match(line):
            add("pytest_annotation", "pytest", number, line)

    if has_pytest_summary:
        for number, line in enumerate(norms, start=1):
            if not line.startswith("E   "):
                continue
            following_is_assertion = number < len(norms) and norms[number].startswith("E   ")
            if not following_is_assertion:
                add("pytest_assertion", "pytest", number, line)

    for number, line in enumerate(norms, start=1):
        if RUFF_RE.match(line):
            add("ruff_lint", "ruff", number, line)
        elif PYRIGHT_RE.match(line):
            add("type_error", "pyright", number, line)
        elif MYPY_RE.match(line):
            add("type_error", "mypy", number, line)
        elif FORMAT_RE.search(line):
            add("format", "ruff", number, line)
        elif any(pattern.search(line) for pattern in DEPENDENCY_PATTERNS):
            add("dependency", dep_tool, number, line)
        elif any(pattern.search(line) for pattern in INFRA_PATTERNS):
            add("infra", "infra", number, line)
        elif any(pattern in line.lower() for pattern in CONFIG_PATTERNS):
            add("config", "actions", number, line)

    traceback = _traceback_line(norms)
    if traceback is not None:
        add("traceback", "generic", traceback[0], traceback[1])

    for number, line in _nontrivial_error_lines(norms):
        if PYTEST_ANNOTATION_RE.match(line):
            continue
        add("error_annotation", "actions", number, line)

    deduped: dict[tuple[str, int, str], tuple[str, str, int, str]] = {}
    for kind, tool, number, text in candidates:
        key = (kind, number, text)
        deduped.setdefault(key, (kind, tool, number, text))
    ordered = sorted(
        deduped.values(),
        key=lambda item: (KIND_PRIORITY.get(item[0], 99), item[2]),
    )[:MAX_EVIDENCE]

    evidence: list[Evidence] = []
    for kind, tool, number, text in ordered:
        digest = hashlib.sha1(f"{kind}:{number}:{text}".encode()).hexdigest()[:8]
        evidence.append(
            Evidence(
                id=f"{kind}-{number}-{digest}",
                kind=kind,
                line_no=number,
                text=text,
                tool=tool,
                in_region=region_start <= number <= region_end,
            )
        )

    return ParsedLog(
        region_start=region_start,
        region_end=region_end,
        evidence=evidence,
        tools_detected=tools,
        has_nontrivial_error=bool(_nontrivial_error_lines(norms)),
    )


__all__ = ["Evidence", "MAX_EVIDENCE", "ParsedLog", "parse_log"]
