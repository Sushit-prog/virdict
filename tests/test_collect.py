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


def _aggregator_log() -> bytes:
    return (
        b"2026-11-19T00:00:00Z lint: success\n"
        b"2026-11-19T00:00:01Z typecheck: success\n"
        b"2026-11-19T00:00:02Z code-quality: failure\n"
        b"2026-11-19T00:00:03Z ##[error]Process completed with exit code 1.\n"
    )


def _real_error_log() -> bytes:
    return (
        b"2026-11-19T00:00:00Z lint: success\n"
        b"2026-11-19T00:00:01Z typecheck: success\n"
        b"2026-11-19T00:00:02Z code-quality: failure\n"
        b"2026-11-19T00:00:03Z ##[error]real failure\n"
    )


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
    assert summary.repos[0].candidates_evaluated == 1
    assert summary.repos[0].excluded == 0
    assert summary.repos[0].skipped_aggregator_only == 0

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
    assert "Candidates evaluated" in rendered
    assert "Excluded" in rendered
    assert "Aggregator-only" in rendered
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
def test_collect_honors_per_repo_with_many_workflows(
    client: GitHubClient,
    tmp_path: Path,
) -> None:
    runs = [run_payload(run_id=run_id, workflow_id=run_id * 10) for run_id in range(1, 7)]
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
        per_repo=3,
        out=tmp_path,
        now=DEFAULT_NOW,
    )

    assert summary.repos[0].runs_saved == 3
    assert summary.repos[0].candidates_evaluated == 3
    assert len(list((tmp_path / "o__r").iterdir())) == 3
    assert jobs_route.call_count == 3


@respx.mock
def test_collect_excludes_default_workflows_by_name_and_path(
    client: GitHubClient,
    tmp_path: Path,
) -> None:
    runs = [
        run_payload(
            run_id=1,
            workflow_id=10,
            name="Duplicate issue check (Codex)",
        ),
        run_payload(
            run_id=2,
            workflow_id=20,
            name="CI",
            workflow_path=".github/workflows/codeql.yml",
        ),
        run_payload(run_id=3, workflow_id=30, name="CI"),
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
        per_repo=1,
        out=tmp_path,
        now=DEFAULT_NOW,
    )

    assert summary.repos[0].runs_saved == 1
    assert summary.repos[0].candidates_evaluated == 1
    assert summary.repos[0].excluded == 2
    assert jobs_route.call_count == 1
    assert not (tmp_path / "o__r" / "1").exists()
    assert not (tmp_path / "o__r" / "2").exists()
    assert (tmp_path / "o__r" / "3").exists()


@respx.mock
def test_collect_excludes_custom_regex_case_insensitively(
    client: GitHubClient,
    tmp_path: Path,
) -> None:
    run = run_payload(
        run_id=1,
        name="Special Workflow",
        workflow_path=".github/workflows/special.yml",
    )
    respx.get(RUNS_URL).mock(return_value=httpx.Response(200, json=runs_response(run)))
    jobs_route = respx.get(url__regex=r"https://api\.github\.com/repos/o/r/actions/runs/\d+/jobs")

    summary = collect_repos(
        client,
        ["o/r"],
        per_repo=1,
        out=tmp_path,
        exclude_patterns=("special",),
        include_default_excludes=False,
        now=DEFAULT_NOW,
    )

    assert summary.repos[0].runs_saved == 0
    assert summary.repos[0].candidates_evaluated == 0
    assert summary.repos[0].excluded == 1
    assert not jobs_route.called


@respx.mock
def test_collect_without_default_excludes_collects_codeql(
    client: GitHubClient,
    tmp_path: Path,
) -> None:
    run = run_payload(
        run_id=1,
        name="CodeQL",
        workflow_path=".github/workflows/codeql.yml",
    )
    respx.get(RUNS_URL).mock(return_value=httpx.Response(200, json=runs_response(run)))
    respx.get("https://api.github.com/repos/o/r/actions/runs/1/jobs").mock(
        return_value=httpx.Response(200, json=jobs_response(job_payload(job_id=101)))
    )
    respx.get("https://api.github.com/repos/o/r/actions/workflows/10/runs").mock(
        return_value=httpx.Response(200, json=runs_response())
    )
    respx.get("https://api.github.com/repos/o/r/actions/jobs/101/logs").mock(
        return_value=httpx.Response(200, content=b"log\n")
    )

    summary = collect_repos(
        client,
        ["o/r"],
        per_repo=1,
        out=tmp_path,
        include_default_excludes=False,
        now=DEFAULT_NOW,
    )

    assert summary.repos[0].runs_saved == 1
    assert summary.repos[0].candidates_evaluated == 1
    assert summary.repos[0].excluded == 0


@respx.mock
def test_collect_skips_aggregator_only_run_by_failed_step(
    client: GitHubClient,
    tmp_path: Path,
) -> None:
    run = run_payload(run_id=1, workflow_id=10, name="lint")
    job = job_payload(
        job_id=501,
        name="lint passed",
        steps=[{"name": "Require every lint job to succeed", "conclusion": "failure"}],
    )
    respx.get(RUNS_URL).mock(return_value=httpx.Response(200, json=runs_response(run)))
    respx.get("https://api.github.com/repos/o/r/actions/runs/1/jobs").mock(
        return_value=httpx.Response(200, json=jobs_response(job))
    )
    success_route = respx.get("https://api.github.com/repos/o/r/actions/workflows/10/runs").mock(
        return_value=httpx.Response(200, json=runs_response())
    )
    respx.get("https://api.github.com/repos/o/r/actions/jobs/501/logs").mock(
        return_value=httpx.Response(200, content=b"log\n")
    )

    summary = collect_repos(
        client,
        ["o/r"],
        per_repo=1,
        out=tmp_path,
        now=DEFAULT_NOW,
    )

    assert summary.repos[0].runs_saved == 0
    assert summary.repos[0].jobs_saved == 0
    assert summary.repos[0].candidates_evaluated == 1
    assert summary.repos[0].skipped_aggregator_only == 1
    assert not (tmp_path / "o__r" / "1").exists()
    assert not success_route.called


@respx.mock
def test_collect_keeps_mixed_run_and_sets_aggregator_flags(
    client: GitHubClient,
    tmp_path: Path,
) -> None:
    run = run_payload(run_id=1, workflow_id=10)
    respx.get(RUNS_URL).mock(return_value=httpx.Response(200, json=runs_response(run)))
    respx.get("https://api.github.com/repos/o/r/actions/runs/1/jobs").mock(
        return_value=httpx.Response(
            200,
            json=jobs_response(
                job_payload(job_id=501, name="summary"),
                job_payload(job_id=502, name="tests"),
            ),
        )
    )
    respx.get("https://api.github.com/repos/o/r/actions/workflows/10/runs").mock(
        return_value=httpx.Response(200, json=runs_response())
    )

    def log_for_job(request: httpx.Request) -> httpx.Response:
        if "/501/" in request.url.path:
            return httpx.Response(200, content=_aggregator_log())
        return httpx.Response(200, content=_real_error_log())

    respx.get(url__regex=r"https://api\.github\.com/repos/o/r/actions/jobs/\d+/logs").mock(
        side_effect=log_for_job
    )

    summary = collect_repos(
        client,
        ["o/r"],
        per_repo=1,
        out=tmp_path,
        now=DEFAULT_NOW,
    )

    assert summary.repos[0].runs_saved == 1
    assert summary.repos[0].jobs_saved == 2
    assert summary.repos[0].candidates_evaluated == 1
    assert summary.repos[0].skipped_aggregator_only == 0
    meta = _read_meta(tmp_path / "o__r" / "1")
    flags = {job["id"]: job["likely_aggregator"] for job in meta["failed_jobs"]}
    assert flags == {501: True, 502: False}


@respx.mock
def test_collect_aggregator_skip_does_not_count_toward_per_repo(
    client: GitHubClient,
    tmp_path: Path,
) -> None:
    runs = [
        run_payload(run_id=1, workflow_id=10, name="lint"),
        run_payload(run_id=2, workflow_id=20),
        run_payload(run_id=3, workflow_id=30),
        run_payload(run_id=4, workflow_id=40),
    ]
    respx.get(RUNS_URL).mock(return_value=httpx.Response(200, json=runs_response(*runs)))

    def jobs_for_run(request: httpx.Request) -> httpx.Response:
        run_id = int(request.url.path.split("/")[-2])
        if run_id == 1:
            job = job_payload(
                job_id=501,
                name="lint passed",
                steps=[
                    {
                        "name": "Require every lint job to succeed",
                        "conclusion": "failure",
                    }
                ],
            )
        else:
            job = job_payload(job_id=run_id + 100)
        return httpx.Response(200, json=jobs_response(job))

    respx.get(url__regex=r"https://api\.github\.com/repos/o/r/actions/runs/\d+/jobs").mock(
        side_effect=jobs_for_run
    )
    respx.get(url__regex=r"https://api\.github\.com/repos/o/r/actions/workflows/\d+/runs").mock(
        return_value=httpx.Response(200, json=runs_response())
    )
    respx.get(url__regex=r"https://api\.github\.com/repos/o/r/actions/jobs/\d+/logs").mock(
        return_value=httpx.Response(200, content=b"log\n")
    )

    summary = collect_repos(
        client,
        ["o/r"],
        per_repo=3,
        out=tmp_path,
        now=DEFAULT_NOW,
    )

    assert summary.repos[0].runs_saved == 3
    assert summary.repos[0].candidates_evaluated == 4
    assert summary.repos[0].skipped_aggregator_only == 1
    assert not (tmp_path / "o__r" / "1").exists()
    assert (tmp_path / "o__r" / "2").exists()
    assert (tmp_path / "o__r" / "3").exists()
    assert (tmp_path / "o__r" / "4").exists()


@respx.mock
def test_collect_stops_after_40_candidates(
    client: GitHubClient,
    tmp_path: Path,
) -> None:
    runs = [
        run_payload(run_id=run_id, workflow_id=run_id * 10, name=f"CI {run_id}")
        for run_id in range(1, 42)
    ]
    respx.get(RUNS_URL).mock(return_value=httpx.Response(200, json=runs_response(*runs)))

    def jobs_for_run(request: httpx.Request) -> httpx.Response:
        run_id = int(request.url.path.split("/")[-2])
        return httpx.Response(
            200,
            json=jobs_response(
                job_payload(
                    job_id=run_id + 100,
                    steps=[{"name": "gate", "conclusion": "failure"}],
                )
            ),
        )

    jobs_route = respx.get(
        url__regex=r"https://api\.github\.com/repos/o/r/actions/runs/\d+/jobs"
    ).mock(side_effect=jobs_for_run)
    respx.get(url__regex=r"https://api\.github\.com/repos/o/r/actions/jobs/\d+/logs").mock(
        return_value=httpx.Response(200, content=b"log\n")
    )

    summary = collect_repos(
        client,
        ["o/r"],
        per_repo=3,
        out=tmp_path,
        now=DEFAULT_NOW,
    )

    assert summary.repos[0].runs_saved == 0
    assert summary.repos[0].candidates_evaluated == 40
    assert summary.repos[0].skipped_aggregator_only == 40
    assert jobs_route.call_count == 40
    assert not (tmp_path / "o__r").exists()


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
