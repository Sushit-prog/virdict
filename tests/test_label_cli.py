from __future__ import annotations

import re
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from citriage.cases import Case, load_cases
from citriage.cli import app
from citriage.label_cli import (
    RAW_ROOT,
    cohens_kappa,
    label_agreement,
    label_relabel,
    load_queue,
    plan_queue,
    render_case,
)
from citriage.labels import Label, LabelName, append_label, latest_labels, load_labels

CASES_PATH = Path("data/cases.jsonl")


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
        labeled_at=datetime(2026, 11, 20, 12, 0, tzinfo=UTC),
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
