from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import respx

from citriage.collect import collect_repos
from citriage.github.client import GitHubClient
from citriage.summary import render_summary
from tests.helpers import (
    DEFAULT_NOW,
    job_payload,
    jobs_response,
    run_payload,
    runs_response,
)

RUNS_URL = "https://api.github.com/repos/o/r/actions/runs"


def _read_meta(path: Path) -> dict[str, Any]:
    return json.loads((path / "meta.json").read_text(encoding="utf-8"))


@respx.mock
def test_collect_writes_redacted_meta_and_logs(
    client: GitHubClient,
    tmp_path: Path,
) -> None:
    token = "test-token"
    github_token = "ghp_" + "a" * 36
    run = run_payload(
        run_id=100,
        workflow_id=10,
        name=f"CI {github_token}",
        head_sha="sha-1",
        pull_requests=[
            {
                "number": 7,
                "base": {"sha": "base-sha", "ref": "main"},
            }
        ],
    )
    job = job_payload(job_id=501, name=f"tests {github_token}")
    raw_log = (
        b"line one\n"
        b"::add-mask::masked-secret\n"
        b"masked-secret\n" + github_token.encode() + b"\n" + token.encode() + b"\nERROR: boom\n"
    )
    success = run_payload(
        run_id=101,
        workflow_id=10,
        head_sha="sha-1",
        conclusion="success",
    )

    respx.get(RUNS_URL).mock(return_value=httpx.Response(200, json=runs_response(run)))
    respx.get("https://api.github.com/repos/o/r/actions/runs/100/jobs").mock(
        return_value=httpx.Response(200, json=jobs_response(job))
    )
    respx.get("https://api.github.com/repos/o/r/actions/workflows/10/runs").mock(
        return_value=httpx.Response(
            200,
            json=runs_response(success),
        )
    )
    blob_url = "https://logs.example/501.txt"
    respx.get("https://api.github.com/repos/o/r/actions/jobs/501/logs").mock(
        return_value=httpx.Response(
            302,
            headers={"Location": blob_url},
        )
    )
    respx.get(blob_url).mock(return_value=httpx.Response(200, content=raw_log))

    summary = collect_repos(
        client,
        ["o/r"],
        per_repo=1,
        out=tmp_path,
        extra_secrets=(token,),
        now=DEFAULT_NOW,
    )

    assert summary.repos[0].runs_saved == 1
    assert summary.repos[0].jobs_saved == 1

    run_path = tmp_path / "o__r" / "100"
    meta = _read_meta(run_path)
    meta_text = (run_path / "meta.json").read_text(encoding="utf-8")
    log_text = (run_path / "job_501.log").read_text(encoding="utf-8")
    assert summary.repos[0].log_sizes == [len(log_text.encode())]

    assert meta["run_id"] == 100
    assert meta["workflow_id"] == 10
    assert meta["workflow_path"] == ".github/workflows/ci.yml"
    assert meta["conclusion"] == "failure"
    assert meta["run_started_at"] is not None
    assert meta["pull_requests"] == [{"number": 7, "base_sha": "base-sha", "base_ref": "main"}]
    assert meta["head_repository"] == {"full_name": "o/r"}
    assert meta["failed_jobs"][0]["id"] == 501
    assert meta["failed_jobs"][0]["failed_step_names"] == ["pytest"]
    assert meta["later_green_same_sha"] is True
    assert meta["logs"][0]["original_bytes_seen"] == len(raw_log)
    assert meta["logs"][0]["kept_bytes"] == len(raw_log)
    assert meta["logs"][0]["written_bytes"] == len(log_text.encode())
    assert meta["logs"][0]["truncated"] is False
    assert meta["logs"][0]["truncation_side"] is None

    assert github_token not in meta_text
    assert token not in meta_text
    assert "masked-secret" not in log_text
    assert github_token not in log_text
    assert token not in log_text
    assert "::add-mask::[REDACTED]" in log_text
    assert "ERROR: boom" in log_text

    rendered = render_summary(summary)
    assert "Repo" in rendered
    assert "Runs saved" in rendered
    assert "Jobs saved" in rendered
    assert "Median log" in rendered
    assert "Max log" in rendered
    assert "Rate limit remaining" in rendered


@respx.mock
def test_collect_is_idempotent(
    client: GitHubClient,
    tmp_path: Path,
) -> None:
    run = run_payload(run_id=100, workflow_id=10, head_sha="sha-1")
    job = job_payload(job_id=501)
    runs_route = respx.get(RUNS_URL).mock(return_value=httpx.Response(200, json=runs_response(run)))
    jobs_route = respx.get("https://api.github.com/repos/o/r/actions/runs/100/jobs").mock(
        return_value=httpx.Response(200, json=jobs_response(job))
    )
    success_route = respx.get("https://api.github.com/repos/o/r/actions/workflows/10/runs").mock(
        return_value=httpx.Response(200, json=runs_response())
    )
    logs_route = respx.get("https://api.github.com/repos/o/r/actions/jobs/501/logs").mock(
        return_value=httpx.Response(200, content=b"log\n")
    )

    first = collect_repos(
        client,
        ["o/r"],
        per_repo=1,
        out=tmp_path,
        now=DEFAULT_NOW,
    )
    run_path = tmp_path / "o__r" / "100"
    meta_before = (run_path / "meta.json").read_text(encoding="utf-8")
    counts_before = (
        jobs_route.call_count,
        success_route.call_count,
        logs_route.call_count,
    )

    second = collect_repos(
        client,
        ["o/r"],
        per_repo=1,
        out=tmp_path,
        now=DEFAULT_NOW,
    )

    assert first.repos[0].runs_saved == 1
    assert second.repos[0].runs_saved == 0
    assert second.repos[0].jobs_saved == 0
    assert second.repos[0].runs_skipped == 1
    assert runs_route.call_count == 2
    assert (
        jobs_route.call_count,
        success_route.call_count,
        logs_route.call_count,
    ) == counts_before
    assert (run_path / "meta.json").read_text(encoding="utf-8") == meta_before


@respx.mock
def test_collect_prefers_distinct_workflows(
    client: GitHubClient,
    tmp_path: Path,
) -> None:
    runs = [
        run_payload(run_id=1, workflow_id=10),
        run_payload(run_id=2, workflow_id=10),
        run_payload(run_id=3, workflow_id=20),
        run_payload(run_id=4, workflow_id=20),
    ]
    respx.get(RUNS_URL).mock(return_value=httpx.Response(200, json=runs_response(*runs)))

    def jobs_for_run(request: httpx.Request) -> httpx.Response:
        run_id = int(request.url.path.split("/")[-2])
        return httpx.Response(
            200,
            json=jobs_response(job_payload(job_id=run_id + 100)),
        )

    jobs_route = respx.get(
        url__regex=r"https://api\.github\.com/repos/o/r/actions/runs/\d+/jobs"
    ).mock(side_effect=jobs_for_run)
    respx.get(url__regex=r"https://api\.github\.com/repos/o/r/actions/workflows/\d+/runs").mock(
        return_value=httpx.Response(200, json=runs_response())
    )
    respx.get(url__regex=r"https://api\.github\.com/repos/o/r/actions/jobs/\d+/logs").mock(
        return_value=httpx.Response(200, content=b"log\n")
    )

    summary = collect_repos(
        client,
        ["o/r"],
        per_repo=2,
        out=tmp_path,
        now=DEFAULT_NOW,
    )

    assert summary.repos[0].runs_saved == 2
    assert (tmp_path / "o__r" / "1").exists()
    assert (tmp_path / "o__r" / "3").exists()
    assert not (tmp_path / "o__r" / "2").exists()
    assert not (tmp_path / "o__r" / "4").exists()
    assert jobs_route.call_count == 2


@respx.mock
def test_collect_records_later_green_false(
    client: GitHubClient,
    tmp_path: Path,
) -> None:
    run = run_payload(run_id=100, workflow_id=10, head_sha="sha-1")
    job = job_payload(job_id=501)
    respx.get(RUNS_URL).mock(return_value=httpx.Response(200, json=runs_response(run)))
    respx.get("https://api.github.com/repos/o/r/actions/runs/100/jobs").mock(
        return_value=httpx.Response(200, json=jobs_response(job))
    )
    respx.get("https://api.github.com/repos/o/r/actions/workflows/10/runs").mock(
        return_value=httpx.Response(200, json=runs_response())
    )
    respx.get("https://api.github.com/repos/o/r/actions/jobs/501/logs").mock(
        return_value=httpx.Response(200, content=b"log\n")
    )

    collect_repos(
        client,
        ["o/r"],
        per_repo=1,
        out=tmp_path,
        now=DEFAULT_NOW,
    )

    meta = _read_meta(tmp_path / "o__r" / "100")
    assert meta["later_green_same_sha"] is False


@respx.mock
def test_collect_records_later_green_null_on_lookup_error(
    client: GitHubClient,
    tmp_path: Path,
) -> None:
    run = run_payload(run_id=100, workflow_id=10, head_sha="sha-1")
    job = job_payload(job_id=501)
    respx.get(RUNS_URL).mock(return_value=httpx.Response(200, json=runs_response(run)))
    respx.get("https://api.github.com/repos/o/r/actions/runs/100/jobs").mock(
        return_value=httpx.Response(200, json=jobs_response(job))
    )
    respx.get("https://api.github.com/repos/o/r/actions/workflows/10/runs").mock(
        return_value=httpx.Response(404)
    )
    respx.get("https://api.github.com/repos/o/r/actions/jobs/501/logs").mock(
        return_value=httpx.Response(200, content=b"log\n")
    )

    collect_repos(
        client,
        ["o/r"],
        per_repo=1,
        out=tmp_path,
        now=DEFAULT_NOW,
    )

    meta = _read_meta(tmp_path / "o__r" / "100")
    assert meta["later_green_same_sha"] is None


@respx.mock
def test_collect_records_head_dropped_truncation(
    client: GitHubClient,
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr("citriage.collect.DEFAULT_KEEP_BYTES", 64)
    monkeypatch.setattr("citriage.collect.DEFAULT_CEILING_BYTES", 4096)
    run = run_payload(run_id=100, workflow_id=10, head_sha="sha-1")
    job = job_payload(job_id=501)
    raw_log = b"x" * 300 + b"\nERROR: final failure\n"
    respx.get(RUNS_URL).mock(return_value=httpx.Response(200, json=runs_response(run)))
    respx.get("https://api.github.com/repos/o/r/actions/runs/100/jobs").mock(
        return_value=httpx.Response(200, json=jobs_response(job))
    )
    respx.get("https://api.github.com/repos/o/r/actions/workflows/10/runs").mock(
        return_value=httpx.Response(200, json=runs_response())
    )
    respx.get("https://api.github.com/repos/o/r/actions/jobs/501/logs").mock(
        return_value=httpx.Response(200, content=raw_log)
    )

    collect_repos(
        client,
        ["o/r"],
        per_repo=1,
        out=tmp_path,
        now=DEFAULT_NOW,
    )

    meta = _read_meta(tmp_path / "o__r" / "100")
    log = meta["logs"][0]
    log_text = (tmp_path / "o__r" / "100" / "job_501.log").read_text(encoding="utf-8")

    assert log["original_bytes_seen"] == len(raw_log)
    assert log["truncated"] is True
    assert log["truncation_side"] == "head_dropped"
    assert log["kept_bytes"] <= 64
    assert "ERROR: final failure" in log_text
    assert meta["truncated_job_ids"] == [501]


@respx.mock
def test_collect_records_ceiling_original_size(
    client: GitHubClient,
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr("citriage.collect.DEFAULT_KEEP_BYTES", 8)
    monkeypatch.setattr("citriage.collect.DEFAULT_CEILING_BYTES", 16)
    run = run_payload(run_id=100, workflow_id=10, head_sha="sha-1")
    job = job_payload(job_id=501)
    respx.get(RUNS_URL).mock(return_value=httpx.Response(200, json=runs_response(run)))
    respx.get("https://api.github.com/repos/o/r/actions/runs/100/jobs").mock(
        return_value=httpx.Response(200, json=jobs_response(job))
    )
    respx.get("https://api.github.com/repos/o/r/actions/workflows/10/runs").mock(
        return_value=httpx.Response(200, json=runs_response())
    )
    respx.get("https://api.github.com/repos/o/r/actions/jobs/501/logs").mock(
        return_value=httpx.Response(200, content=b"z" * 100)
    )

    collect_repos(
        client,
        ["o/r"],
        per_repo=1,
        out=tmp_path,
        now=DEFAULT_NOW,
    )

    meta = _read_meta(tmp_path / "o__r" / "100")
    log = meta["logs"][0]
    assert log["original_bytes_seen"] == "50MiB+"
    assert log["truncated"] is True
    assert log["truncation_side"] == "head_dropped"


@respx.mock
def test_collect_stops_cleanly_on_rate_limit(
    client: GitHubClient,
    tmp_path: Path,
) -> None:
    run = run_payload(run_id=100, workflow_id=10, head_sha="sha-1")
    future = int((datetime.now(UTC) + timedelta(hours=1)).timestamp())
    respx.get(RUNS_URL).mock(return_value=httpx.Response(200, json=runs_response(run)))
    jobs_route = respx.get("https://api.github.com/repos/o/r/actions/runs/100/jobs").mock(
        return_value=httpx.Response(
            403,
            json={"message": "API rate limit exceeded"},
            headers={
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(future),
            },
        )
    )
    logs_route = respx.get("https://api.github.com/repos/o/r/actions/jobs/501/logs")

    summary = collect_repos(
        client,
        ["o/r"],
        per_repo=1,
        out=tmp_path,
        now=DEFAULT_NOW,
    )

    assert summary.stopped_reason is not None
    assert "rate limit" in summary.stopped_reason.lower()
    assert summary.rate_limit_remaining == 0
    assert summary.repos[0].runs_saved == 0
    assert jobs_route.called
    assert not logs_route.called
