from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from citriage.bench.systems import Prediction
from citriage.cases import Case
from citriage.labels import Label, LabelAction, LabelName
from citriage.parse.logparse import Evidence

DEFAULT_NOW = datetime(2026, 11, 19, 12, 0, tzinfo=UTC)


def iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def run_payload(
    *,
    run_id: int = 1,
    workflow_id: int | None = 10,
    name: str = "CI",
    workflow_path: str | None = ".github/workflows/ci.yml",
    head_sha: str = "sha-1",
    conclusion: str = "failure",
    created_at: datetime | None = None,
    run_attempt: int = 1,
    event: str = "push",
    pull_requests: list[dict[str, Any]] | None = None,
    head_repository: str | None = "o/r",
) -> dict[str, Any]:
    created = created_at or DEFAULT_NOW - timedelta(days=1)
    return {
        "id": run_id,
        "name": name,
        "workflow_id": workflow_id,
        "workflow_path": workflow_path,
        "event": event,
        "conclusion": conclusion,
        "head_sha": head_sha,
        "head_branch": "main",
        "run_attempt": run_attempt,
        "run_started_at": iso(created),
        "created_at": iso(created),
        "html_url": f"https://github.com/o/r/actions/runs/{run_id}",
        "pull_requests": pull_requests or [],
        "head_repository": ({"full_name": head_repository} if head_repository else None),
    }


def job_payload(
    *,
    job_id: int = 101,
    name: str = "tests",
    conclusion: str = "failure",
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
    steps: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    started = started_at or DEFAULT_NOW - timedelta(hours=2)
    completed = completed_at or DEFAULT_NOW - timedelta(hours=1)
    return {
        "id": job_id,
        "name": name,
        "conclusion": conclusion,
        "started_at": iso(started),
        "completed_at": iso(completed),
        "steps": steps
        if steps is not None
        else [
            {"name": "setup", "conclusion": "success"},
            {"name": "pytest", "conclusion": "failure"},
        ],
    }


def runs_response(*runs: dict[str, Any]) -> dict[str, Any]:
    return {"total_count": len(runs), "workflow_runs": list(runs)}


def jobs_response(*jobs: dict[str, Any]) -> dict[str, Any]:
    return {"total_count": len(jobs), "jobs": list(jobs)}


def bench_evidence(
    *,
    id: str = "evidence-1",
    kind: str = "test_failure",
    line_no: int = 10,
    text: str = "FAILED tests/test_x.py::test_a",
) -> Evidence:
    return Evidence(
        id=id,
        kind=kind,
        line_no=line_no,
        text=text,
        tool="pytest",
        in_region=True,
    )


def bench_case(
    *,
    case_id: str = "c1",
    repo: str = "o/r",
    split: Literal["dev", "test"] = "dev",
    in_benchmark: bool = True,
    evidence: list[Evidence] | None = None,
) -> Case:
    return Case(
        case_id=case_id,
        repo=repo,
        run_id=1,
        run_dir_name="1",
        head_sha="sha-1",
        created_at=DEFAULT_NOW,
        workflow_name="CI",
        step="Run tests",
        job_id=100,
        job_name="job",
        sibling_job_ids=[],
        signature="sig",
        signature_rank=1,
        likely_aggregator=False,
        flake_confirmed=False,
        in_benchmark=in_benchmark,
        split=split,
        seed="seed",
        region_start=1,
        region_end=100,
        tools_detected=["pytest"],
        has_nontrivial_error=False,
        evidence=evidence or [],
        b0_label=LabelName.UNKNOWN,
        b0_culprit_file=None,
        b0_evidence_ids=[],
    )


def bench_label(
    *,
    case_id: str = "c1",
    labeler: str = "alice",
    action: LabelAction = "label",
    label: LabelName | None = LabelName.TEST_FAILURE,
    evidence_lines: list[int] | None = None,
    culprit_file: str | None = None,
    labeled_at: datetime | None = None,
    id: str = "label-1",
    supersedes: str | None = None,
    repo: str = "o/r",
) -> Label:
    return Label(
        id=id,
        case_id=case_id,
        repo=repo,
        run_id=1,
        job_id=100,
        action=action,
        label=label,
        evidence_lines=evidence_lines or [],
        culprit_file=culprit_file,
        labeler=labeler,
        labeled_at=labeled_at or DEFAULT_NOW,
        supersedes=supersedes,
    )


def bench_prediction(
    *,
    label: LabelName = LabelName.TEST_FAILURE,
    culprit_file: str | None = None,
    cited_evidence_ids: list[str] | None = None,
) -> Prediction:
    return Prediction(
        label=label,
        culprit_file=culprit_file,
        cited_evidence_ids=cited_evidence_ids or [],
    )
