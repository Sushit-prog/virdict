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
    "Cleaning up orphan processes",
    "Post Run",
)

PYTEST_SUMMARY_RE = re.compile(r"^(FAILED|ERROR)\s+(\S+::\S+)\s+-\s+(.*)$")
PYTEST_COUNT_RE = re.compile(r"=+\s*\d+\s+failed")
PYTEST_ANNOTATION_RE = re.compile(r"^##\[error\]\s*\S+::\S+")
RUFF_RE = re.compile(r"^(.+?):(\d+):(\d+): ([A-Z]+\d+) (.+)$")
RUFF_SUMMARY_RE = re.compile(r"^Found \d+ errors?\.?$")
MYPY_RE = re.compile(r"^(.+?):(\d+):(?:\d+:)? error: (.*)$")
MYPY_SUMMARY_RE = re.compile(r"^Found \d+ errors? in \d+ files?")
PYRIGHT_RE = re.compile(r"^(.+?):(\d+):(\d+) - (?:error|warning): (.*)$")
FORMAT_RE = re.compile(r"would reformat (.+)$", re.IGNORECASE)
ACTIONLINT_RE = re.compile(r"^(.+?):(\d+):(\d+): (.+?) \[([A-Za-z0-9_-]+)\]$")
PRE_COMMIT_HOOK_RE = re.compile(r"^-\s+hook id:\s*(.+)$")
PRE_COMMIT_FAIL_RE = re.compile(
    r"^(?:Failed|Files were modified by this hook)(?:\s*[-:].*)?$", re.IGNORECASE
)
RATCHET_FAIL_RE = re.compile(
    r"^FAIL: (strict-rule totals|basedpyright errors) (?:grew past|exceeded)", re.IGNORECASE
)
RATCHET_COUNT_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*): total \d+ over ceiling \d+ \(this change added \d+\)"
)
RATCHET_PATH_RE = re.compile(r"^(.+?):(\d+)$")
TEST_RUNNER_IMPORT_RE = re.compile(r"^ImportError while loading conftest ['\"](.+?)['\"]\.?$")
TEST_RUNNER_E_RE = re.compile(r"^E\s{3}(.+)$")
TEST_FRAME_RE = re.compile(r"^(.+?):(\d+): in .+$")
FATAL_PYTHON_RE = re.compile(r"^Fatal Python error:\s*(.+)$")
PYTHON_STACK_FRAME_RE = re.compile(r'^\s*File "(.+?)", line (\d+) in (.+)$')
TOX_FAIL_RE = re.compile(r"^\s*\S+:\s+FAIL code \d+")
TOX_EVALUATION_RE = re.compile(r"^\s*evaluation failed\s*:\(")
UV_INCOMPATIBILITY_RE = re.compile(
    r"^The package `(.+?)` requires `(.+)`, but `(.+?)` is installed\.?$"
)
BUS_ERROR_RE = re.compile(r"\b(?:Bus error|Segmentation fault)\b", re.IGNORECASE)

DEPENDENCY_PATTERNS = (
    re.compile(r"^ERROR:"),
    re.compile(r"^error: No solution found"),
    re.compile(r"Could not find a version"),
    re.compile(r"SolverProblemError"),
    re.compile(r"Failed building wheel"),
    UV_INCOMPATIBILITY_RE,
    re.compile(r"^Found \d+ incompatibilit(?:y|ies)\b"),
)
INFRA_PATTERNS = (
    BUS_ERROR_RE,
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
    "tox": re.compile(r"\btox\b"),
    "ruff": re.compile(r"\bruff\b"),
    "mypy": re.compile(r"\bmypy\b"),
    "pyright": re.compile(r"\bpyright\b"),
    "basedpyright": re.compile(r"\bbasedpyright\b"),
    "actionlint": re.compile(r"\bactionlint\b"),
    "pre-commit": re.compile(r"\bpre-commit\b"),
    "uv": re.compile(r"\buv\b"),
    "poetry": re.compile(r"\bpoetry\b"),
    "pip": re.compile(r"\bpip\b"),
}

KIND_PRIORITY = {
    "config": 0,
    "infra": 1,
    "dependency": 2,
    "ratchet": 3,
    "lint_format": 4,
    "format": 4,
    "ruff_lint": 5,
    "type_error": 6,
    "test_failure": 7,
    "pytest_summary": 8,
    "pytest_count": 9,
    "pytest_assertion": 10,
    "pytest_annotation": 11,
    "traceback": 12,
    "error_annotation": 13,
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


def _tail_window(norms: list[str], region_end: int, cleanup_index: int) -> list[tuple[int, str]]:
    end = cleanup_index
    if region_end:
        end = min(end, max(0, region_end - 1))
    tail: list[tuple[int, str]] = []
    for index in range(end - 1, -1, -1):
        text = norms[index].strip()
        if not text:
            continue
        if text.startswith(CLEANUP_PREFIXES) or EXIT_CODE_RE.match(text):
            continue
        tail.append((index + 1, text[:MAX_TEXT]))
        if len(tail) == 15:
            break
    tail.reverse()
    return tail


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
    has_fatal_python = any(FATAL_PYTHON_RE.match(line.strip()) for line in norms)
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

    ratchet_tool: str | None = None
    for number, line in enumerate(norms, start=1):
        stripped = line.strip()
        if stripped.startswith("##[group]Run") or stripped.startswith("##[endgroup]"):
            ratchet_tool = None

        ratchet_fail = RATCHET_FAIL_RE.match(stripped)
        if ratchet_fail is not None:
            name = ratchet_fail.group(1).lower()
            ratchet_tool = "ruff" if name.startswith("strict-rule") else "basedpyright"
            add("ratchet", "ratchet", number, stripped)
            continue

        if ratchet_tool is not None:
            if RATCHET_COUNT_RE.match(stripped) is not None:
                kind = "lint_format" if ratchet_tool == "ruff" else "type_error"
                add(kind, ratchet_tool, number, stripped)
                continue
            if RATCHET_PATH_RE.match(stripped) is not None:
                kind = "lint_format" if ratchet_tool == "ruff" else "type_error"
                add(kind, ratchet_tool, number, stripped)
                continue
            if "BREACHED RULES:" in stripped:
                add("ratchet", "ratchet", number, stripped)
                continue
            if not stripped:
                ratchet_tool = None

    for number, line in enumerate(norms, start=1):
        stripped = line.strip()
        if RUFF_SUMMARY_RE.match(stripped):
            add("lint_format", "ruff", number, stripped)
        elif MYPY_SUMMARY_RE.match(stripped):
            add("type_error", "mypy", number, stripped)
        elif ACTIONLINT_RE.match(stripped):
            add("lint_format", "actionlint", number, stripped)
        elif PRE_COMMIT_HOOK_RE.match(stripped) or PRE_COMMIT_FAIL_RE.match(stripped):
            add("lint_format", "pre-commit", number, stripped)
        elif TOX_FAIL_RE.match(stripped) or TOX_EVALUATION_RE.match(stripped):
            add("test_failure", "tox", number, stripped)
        elif (
            has_fatal_python
            and "pytest" in tools
            and (FATAL_PYTHON_RE.match(stripped) or PYTHON_STACK_FRAME_RE.match(stripped))
        ) or (
            TEST_RUNNER_IMPORT_RE.match(stripped)
            or TEST_FRAME_RE.match(stripped)
            or (TEST_RUNNER_E_RE.match(stripped) and not has_pytest_summary)
        ):
            add("test_failure", "pytest", number, stripped)

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

    if not any(item.in_region for item in evidence):
        for number, text in _tail_window(norms, region_end, _cleanup_index(norms)):
            digest = hashlib.sha1(f"tail:{number}:{text}".encode()).hexdigest()[:8]
            evidence.append(
                Evidence(
                    id=f"tail-{number}-{digest}",
                    kind="tail",
                    line_no=number,
                    text=text,
                    tool="unknown",
                    in_region=True,
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
