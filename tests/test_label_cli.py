from __future__ import annotations

import ast
import re
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TextIO, cast

import pytest
from typer.testing import CliRunner

from citriage.cases import Case, load_cases
from citriage.cli import app
from citriage.label_cli import (
    RAW_ROOT,
    _discard_pending_input,
    _normalize_command,
    _prompt,
    cohens_kappa,
    label_agreement,
    label_relabel,
    load_queue,
    plan_queue,
    render_case,
)
from citriage.labels import Label, LabelName, append_label, latest_labels, load_labels
from citriage.parse.logparse import Evidence
from citriage.storage import log_path

CASES_PATH = Path("data/cases.jsonl")


def _write_log(case: Case, raw_root: Path, lines: list[str]) -> Path:
    path = log_path(
        raw_root,
        case.repo,
        case.run_id,
        case.job_id,
        run_dir_name=case.run_dir_name or None,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_case_queue(tmp_path: Path, case: Case) -> tuple[Path, Path]:
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(case.model_dump_json() + "\n", encoding="utf-8")
    queue_path = tmp_path / "queue.json"
    plan_queue(
        [case],
        queue_path,
        split_targets={"dev": 1, "test": 1},
        flake_minimum=0,
        tool_minimum=0,
    )
    return cases_path, queue_path


def _case(case_id: str, repo: str, split: str, rank: int, flake: bool, tools: list[str]) -> Case:
    now = datetime(2026, 11, 20, 12, 0, tzinfo=UTC)
    return Case(
        case_id=case_id,
        repo=repo,
        run_id=1,
        run_dir_name="1",
        head_sha="abc123",
        created_at=now,
        workflow_name="CI",
        html_url=f"https://github.com/{repo}/actions/runs/1",
        step="Run tests",
        failed_step_names=["Run tests"],
        job_id=100,
        job_name="job",
        sibling_job_ids=[],
        signature="sig",
        signature_rank=rank,
        likely_aggregator=False,
        flake_confirmed=flake,
        in_benchmark=True,
        exclusion_reason="none",
        split=split,
        seed="seed",
        region_start=1,
        region_end=2,
        tools_detected=tools,
        has_nontrivial_error=True,
        evidence=[],
        b0_label=LabelName.UNKNOWN,
        b0_culprit_file=None,
        b0_evidence_ids=[],
    )


def test_plan_queue_respects_constraints_and_reports_relaxations(tmp_path: Path) -> None:
    cases = load_cases(CASES_PATH)
    by_id = {case.case_id: case for case in cases}
    path = tmp_path / "queue.json"

    plan = plan_queue(cases, path)

    assert all(by_id[entry.case_id].in_benchmark for entry in plan.entries)
    assert all(
        entry.signature_rank is not None and entry.signature_rank <= 3 for entry in plan.entries
    )
    repo_counts: Counter[tuple[str, str]] = Counter(
        (entry.repo, entry.split) for entry in plan.entries
    )
    assert max(repo_counts.values()) <= 15
    splits = Counter(entry.split for entry in plan.entries)
    assert splits["dev"] == 30
    assert splits["test"] <= 60
    assert sum(entry.flake_confirmed for entry in plan.entries) >= 15
    tool_cases = sum(
        bool(set(entry.tools_detected) & {"ruff", "mypy", "pip"}) for entry in plan.entries
    )
    assert tool_cases >= 10
    assert plan.relaxations == [
        "signature_rank from <=2 to <=3",
        "per-repo cap from 12 to 15",
    ]
    assert plan.shortfalls == ["test short by 2"]
    assert load_queue(path) == plan


def test_plan_never_includes_excluded_cases(tmp_path: Path) -> None:
    case = _case("c1", "o/r", "dev", 1, False, ["ruff"])
    case.in_benchmark = False
    case.exclusion_reason = "aggregator"

    plan = plan_queue(
        [case],
        tmp_path / "queue.json",
        split_targets={"dev": 1, "test": 1},
        flake_minimum=0,
        tool_minimum=0,
    )

    assert plan.entries == []


def test_render_case_hides_b0_and_flake_fields() -> None:
    case = load_cases(CASES_PATH)[0]

    text = render_case(case, RAW_ROOT)

    assert "b0_label" not in text
    assert "flake_confirmed" not in text
    assert "flaky" not in text
    assert "raw log:" in text
    assert "html_url:" in text
    assert "failed steps:" in text
    assert any(" | " in line for line in text.splitlines())


def test_more_context_shows_more_log_lines() -> None:
    case = load_cases(CASES_PATH)[0]
    pattern = re.compile(r"^\s*\d+ \|", re.MULTILINE)

    short = render_case(case, RAW_ROOT, context_lines=40)
    long = render_case(case, RAW_ROOT, context_lines=150)

    assert len(pattern.findall(long)) > len(pattern.findall(short))


def test_label_cli_does_not_import_parser_or_b0() -> None:
    source_path = Path(__file__).parents[1] / "src" / "citriage" / "label_cli.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)

    forbidden = [
        module
        for module in modules
        if module == "citriage.parser"
        or module.startswith("citriage.parser.")
        or module == "citriage.parse"
        or module.startswith("citriage.parse.")
        or "b0" in module
        or "baselines" in module
    ]

    assert forbidden == []


def test_render_case_pins_failed_lines_before_context(tmp_path: Path) -> None:
    case = _case("c1", "o/r", "dev", 1, False, [])
    long_failure = "FAILED tests/test_x.py::test_y - " + "z" * 400
    summary_header = (
        "\x1b[36m\x1b[1m=========================== short test summary info "
        "============================\x1b[0m"
    )
    final_summary = (
        "2026-11-21T00:00:02.0000000Z =========== 1 failed, 2403 passed, "
        "459 warnings in 81.81s (0:01:21) ============"
    )
    lines = [
        "2026-11-21T00:00:00.0000000Z setup",
        "2026-11-21T00:00:01.0000000Z [gw0] [ 10%] FAILED tests/progress.py::test_p",
        summary_header,
        long_failure,
        final_summary,
        "2026-11-21T00:00:03.0000000Z ##[error]Process completed with exit code 1.",
    ]
    _write_log(case, tmp_path, lines)

    text = render_case(case, tmp_path)

    assert "\x1b[" not in text
    summary = text.index("short test summary info")
    failed = text.index("FAILED tests/test_x.py::test_y")
    total = text.index("1 failed, 2403 passed")
    context = text.index("log context chunk 0")
    assert summary < failed < total < context
    rendered_failure = next(
        line for line in text.splitlines() if "FAILED tests/test_x.py::test_y" in line
    )
    raw = rendered_failure.split(" | ", 1)[1]
    assert len(raw) <= 300
    assert raw.endswith("...")


def test_render_case_excludes_parser_evidence_and_predictions() -> None:
    case = _case("c1", "o/r", "dev", 1, False, [])
    case.evidence = [
        Evidence(
            id="parser-secret-id",
            kind="pytest",
            line_no=7,
            text="PARSER_EVIDENCE_SECRET",
            tool="pytest",
            in_region=True,
        )
    ]
    case.b0_label = LabelName.TYPE_ERROR
    case.b0_culprit_file = "b0-secret.py"
    case.b0_evidence_ids = ["b0-secret-id"]

    text = render_case(case, RAW_ROOT)

    for marker in (
        "PARSER_EVIDENCE_SECRET",
        "parser-secret-id",
        "b0-secret.py",
        "b0-secret-id",
        "b0_label",
        "b0_culprit_file",
        "b0_evidence_ids",
        "kind=",
    ):
        assert marker not in text
    assert "prediction" not in text.lower()


def test_normalize_command_handles_whitespace_and_case() -> None:
    assert _normalize_command("  m \n") == "M"


def test_prompt_uses_first_line_and_discards_pending(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bool] = []
    monkeypatch.setattr("builtins.input", lambda label: "  m\nextra\n")
    monkeypatch.setattr("citriage.label_cli._discard_pending_input", lambda: calls.append(True))

    assert _prompt("> ") == "m"
    assert calls == [True]


def test_discard_pending_input_is_noop_for_non_tty() -> None:
    class Stream:
        def isatty(self) -> bool:
            return False

        def readline(self) -> str:
            raise AssertionError("readline should not be called")

    _discard_pending_input(cast(TextIO, Stream()))


def test_label_next_m_prints_earlier_chunk_and_start_notice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case("c1", "o/r", "dev", 1, False, [])
    lines = [f"line {number:03d}" for number in range(1, 101)]
    lines.append("##[error]Process completed with exit code 1.")
    _write_log(case, tmp_path, lines)
    cases_path, queue_path = _write_case_queue(tmp_path, case)
    answers = iter(["m", "m", "m", "q"])
    monkeypatch.setattr("citriage.label_cli._prompt", lambda label: next(answers))

    result = CliRunner().invoke(
        app,
        [
            "label",
            "next",
            "--cases",
            str(cases_path),
            "--queue",
            str(queue_path),
            "--labels",
            str(tmp_path / "labels.jsonl"),
            "--raw",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "log context chunk 0" in result.output
    assert "log context chunk 1" in result.output
    assert "log context chunk 2" in result.output
    assert "start of log reached" in result.output
    assert result.output.count("line 001") == 1


def test_invalid_command_does_not_rerender_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case("c1", "o/r", "dev", 1, False, [])
    cases_path, queue_path = _write_case_queue(tmp_path, case)
    render_calls: list[dict[str, object]] = []

    def fake_render_case(*_args: object, **kwargs: object) -> str:
        render_calls.append(kwargs)
        return "CASE VIEW"

    answers = iter([" bad ", "Q"])
    monkeypatch.setattr("citriage.label_cli.render_case", fake_render_case)
    monkeypatch.setattr("citriage.label_cli._prompt", lambda label: next(answers))

    result = CliRunner().invoke(
        app,
        [
            "label",
            "next",
            "--cases",
            str(cases_path),
            "--queue",
            str(queue_path),
            "--labels",
            str(tmp_path / "labels.jsonl"),
            "--raw",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert len(render_calls) == 1
    assert result.output.count("CASE VIEW") == 1
    assert "Invalid command. Valid: 1-7, M, S, U, Q." in result.output


def test_label_next_evidence_is_labeler_entered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case("c1", "o/r", "dev", 1, True, [])
    case.evidence = [
        Evidence(
            id="parser-secret-id",
            kind="pytest",
            line_no=7,
            text="PARSER_EVIDENCE_SECRET",
            tool="pytest",
            in_region=True,
        )
    ]
    cases_path, queue_path = _write_case_queue(tmp_path, case)
    labels_path = tmp_path / "labels.jsonl"
    answers = iter(["1", "", "", "", "manual note"])
    monkeypatch.setattr("citriage.label_cli._prompt", lambda label: next(answers))

    result = CliRunner().invoke(
        app,
        [
            "label",
            "next",
            "--cases",
            str(cases_path),
            "--queue",
            str(queue_path),
            "--labels",
            str(labels_path),
            "--raw",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    records = load_labels(labels_path)
    assert len(records) == 1
    assert records[0].evidence_lines == []
    assert records[0].culprit_file is None
    assert "PARSER_EVIDENCE_SECRET" not in result.output


def test_test_split_requires_confirm_test() -> None:
    result = CliRunner().invoke(app, ["label", "next", "--split", "test"])

    assert result.exit_code == 2
    assert "must not drive parser or B0 changes" in result.output
    assert "--confirm-test" in result.output


def test_cohens_kappa_known_example() -> None:
    first = [
        LabelName.TEST_FAILURE,
        LabelName.TEST_FAILURE,
        LabelName.LINT_FORMAT,
        LabelName.LINT_FORMAT,
    ]
    second = [
        LabelName.TEST_FAILURE,
        LabelName.LINT_FORMAT,
        LabelName.LINT_FORMAT,
        LabelName.LINT_FORMAT,
    ]

    assert cohens_kappa(first, second) == pytest.approx(0.5)


def test_relabel_creates_pending_r2_record_and_agreement_uses_it(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    labels_path = tmp_path / "labels.jsonl"
    base = Label(
        case_id="c1",
        repo="o/r",
        run_id=1,
        job_id=100,
        label=LabelName.TEST_FAILURE,
        labeler="alice",
        labeled_at=datetime.now(UTC),
    )
    append_label(base, labels_path)

    label_relabel(n=1, labels_path=labels_path)
    records = load_labels(labels_path)
    pending = records[-1]

    assert pending.action == "pending"
    assert pending.label is None
    assert pending.labeler == "alice-r2"
    assert pending.supersedes == base.id
    assert latest_labels(records) == {("c1", "alice"): base}

    second = Label(
        case_id="c1",
        repo="o/r",
        run_id=1,
        job_id=100,
        label=LabelName.TEST_FAILURE,
        labeler="alice-r2",
        labeled_at=base.labeled_at + timedelta(seconds=1),
    )
    append_label(second, labels_path)

    label_agreement("alice", labels_path=labels_path)
    output = capsys.readouterr().out

    assert "cohen's kappa: 1.000" in output
    assert "shared cases: 1" in output
