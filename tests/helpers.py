from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

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
