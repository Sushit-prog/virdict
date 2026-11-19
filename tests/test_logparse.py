from pathlib import Path

from citriage.parse.logparse import parse_log

FIXTURES = Path(__file__).parent / "fixtures" / "logs"


def _load(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_pytest_failure_exact_lines() -> None:
    parsed = parse_log(_load("pytest_failure.log"))
    assert (parsed.region_start, parsed.region_end) == (1, 12)
    assert "pytest" in parsed.tools_detected
    assert not parsed.has_nontrivial_error
    by_kind = {item.kind: item for item in parsed.evidence}
    assert by_kind["pytest_summary"].line_no == 10
    assert by_kind["pytest_assertion"].line_no == 8
    assert by_kind["pytest_count"].line_no == 11
    assert all(item.in_region for item in parsed.evidence)
    assert all(len(item.text) <= 300 for item in parsed.evidence)


def test_ruff_exact_line() -> None:
    parsed = parse_log(_load("ruff.log"))
    assert (parsed.region_start, parsed.region_end) == (1, 4)
    lint = next(item for item in parsed.evidence if item.kind == "ruff_lint")
    assert lint.line_no == 2
    assert lint.tool == "ruff"


def test_mypy_exact_line() -> None:
    parsed = parse_log(_load("mypy.log"))
    item = next(item for item in parsed.evidence if item.kind == "type_error")
    assert item.line_no == 2
    assert item.tool == "mypy"


def test_uv_resolve_failure() -> None:
    parsed = parse_log(_load("uv_resolve.log"))
    item = next(item for item in parsed.evidence if item.kind == "dependency")
    assert item.line_no == 3
    assert item.tool == "uv"
    assert "uv" in parsed.tools_detected


def test_timeout_without_exit_code_uses_fallback_window() -> None:
    parsed = parse_log(_load("timeout.log"))
    assert (parsed.region_start, parsed.region_end) == (1, 4)
    infra = [item for item in parsed.evidence if item.kind == "infra"]
    assert [item.line_no for item in infra] == [3, 4]
    assert all(item.in_region for item in infra)


def test_cancelled_without_cleanup_or_exit_code() -> None:
    parsed = parse_log(_load("cancelled.log"))
    assert (parsed.region_start, parsed.region_end) == (1, 3)
    assert not parsed.has_nontrivial_error
    assert [item.line_no for item in parsed.evidence if item.kind == "infra"] == [2]


def test_error_only_log() -> None:
    parsed = parse_log(_load("error_only.log"))
    assert (parsed.region_start, parsed.region_end) == (1, 3)
    assert parsed.has_nontrivial_error
    item = next(item for item in parsed.evidence if item.kind == "error_annotation")
    assert item.line_no == 2


def test_no_exit_code_line() -> None:
    parsed = parse_log(_load("no_exit_code.log"))
    assert (parsed.region_start, parsed.region_end) == (1, 2)
    assert parsed.has_nontrivial_error


def test_ansi_and_timestamps_are_stripped() -> None:
    parsed = parse_log(_load("ansi.log"))
    assert (parsed.region_start, parsed.region_end) == (1, 3)
    item = next(item for item in parsed.evidence if item.kind == "ruff_lint")
    assert item.line_no == 2
    assert item.text.startswith("src/x.py:3:1")
    assert "\x1b" not in item.text
    assert "2026-" not in item.text


def test_5000_line_noise_keeps_exact_numbers() -> None:
    parsed = parse_log(_load("noise_5000.log"))
    assert (parsed.region_start, parsed.region_end) == (4997, 4999)
    item = next(item for item in parsed.evidence if item.kind == "pytest_summary")
    assert item.line_no == 4998
    assert item.in_region


def test_evidence_is_capped() -> None:
    lines = ["##[group]Run echo noise"]
    lines.extend(f"##[error]failure number {number}" for number in range(1, 60))
    lines.append("##[error]Process completed with exit code 1.")
    lines.append("##[group]Post job cleanup.")
    parsed = parse_log("\n".join(lines) + "\n")
    assert len(parsed.evidence) <= 40
