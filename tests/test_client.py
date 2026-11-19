from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import pytest
import respx

from citriage.errors import RateLimitExhausted
from citriage.github.client import GitHubClient
from citriage.models import WorkflowRun
from tests.helpers import DEFAULT_NOW, job_payload, jobs_response, run_payload, runs_response

RUNS_URL = "https://api.github.com/repos/o/r/actions/runs"


@respx.mock
def test_list_failed_runs_paginates_and_filters(
    client: GitHubClient,
) -> None:
    page_one = runs_response(
        run_payload(
            run_id=1,
            created_at=DEFAULT_NOW - timedelta(days=1),
        ),
        run_payload(
            run_id=2,
            created_at=DEFAULT_NOW - timedelta(days=90),
        ),
    )
    page_two = runs_response(
        run_payload(
            run_id=3,
            created_at=DEFAULT_NOW - timedelta(days=2),
        )
    )
    route = respx.get(RUNS_URL).mock(
        side_effect=[
            httpx.Response(
                200,
                json=page_one,
                headers={
                    "Link": (
                        "<https://api.github.com/repos/o/r/actions/runs"
                        '?status=failure&per_page=50&page=2>; rel="next"'
                    )
                },
            ),
            httpx.Response(200, json=page_two),
        ]
    )

    runs = client.list_failed_runs(
        "o",
        "r",
        since=DEFAULT_NOW - timedelta(days=80),
    )

    assert [run.id for run in runs] == [1, 3]
    assert route.call_count == 2
    assert route.calls[0].request.url.params["status"] == "failure"
    assert route.calls[0].request.url.params["per_page"] == "50"


@respx.mock
def test_list_failed_runs_stops_when_page_is_entirely_old(
    client: GitHubClient,
) -> None:
    old_page = runs_response(
        run_payload(
            run_id=1,
            created_at=DEFAULT_NOW - timedelta(days=100),
        )
    )
    route = respx.get(RUNS_URL).mock(
        return_value=httpx.Response(
            200,
            json=old_page,
            headers={
                "Link": ('<https://api.github.com/repos/o/r/actions/runs?page=2>; rel="next"')
            },
        )
    )

    runs = client.list_failed_runs(
        "o",
        "r",
        since=DEFAULT_NOW - timedelta(days=80),
    )

    assert runs == []
    assert route.call_count == 1


@respx.mock
def test_list_failed_jobs_paginates_and_filters(
    client: GitHubClient,
) -> None:
    jobs_url = "https://api.github.com/repos/o/r/actions/runs/1/jobs"
    route = respx.get(jobs_url).mock(
        side_effect=[
            httpx.Response(
                200,
                json=jobs_response(
                    job_payload(job_id=101),
                    job_payload(job_id=102, conclusion="success"),
                ),
                headers={"Link": f'<{jobs_url}?page=2>; rel="next"'},
            ),
            httpx.Response(
                200,
                json=jobs_response(job_payload(job_id=103)),
            ),
        ]
    )

    jobs = client.list_failed_jobs("o", "r", 1)

    assert [job.id for job in jobs] == [101, 103]
    assert route.call_count == 2


@respx.mock
def test_rate_limit_stops_before_next_request(client: GitHubClient) -> None:
    future = int((datetime.now(UTC) + timedelta(hours=1)).timestamp())
    respx.get(RUNS_URL).mock(
        return_value=httpx.Response(
            200,
            json=runs_response(),
            headers={
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(future),
            },
        )
    )
    jobs_route = respx.get("https://api.github.com/repos/o/r/actions/runs/1/jobs")

    assert (
        client.list_failed_runs(
            "o",
            "r",
            since=DEFAULT_NOW - timedelta(days=80),
        )
        == []
    )

    with pytest.raises(RateLimitExhausted):
        client.list_failed_jobs("o", "r", 1)

    assert not jobs_route.called


@respx.mock
def test_5xx_is_retried_with_bounded_backoff() -> None:
    sleeps: list[float] = []
    with GitHubClient(
        "test-token",
        sleep=sleeps.append,
        retry_attempts=3,
    ) as client:
        route = respx.get(RUNS_URL).mock(
            side_effect=[
                httpx.Response(502),
                httpx.Response(503),
                httpx.Response(200, json=runs_response()),
            ]
        )

        assert (
            client.list_failed_runs(
                "o",
                "r",
                since=DEFAULT_NOW - timedelta(days=80),
            )
            == []
        )

    assert route.call_count == 3
    assert len(sleeps) == 2


@respx.mock
def test_log_download_follows_redirect_without_forwarding_auth(
    client: GitHubClient,
) -> None:
    api_url = "https://api.github.com/repos/o/r/actions/jobs/101/logs"
    blob_url = "https://logs.example/101.txt"
    seen_auth: list[str | None] = []

    def blob_response(request: httpx.Request) -> httpx.Response:
        seen_auth.append(request.headers.get("authorization"))
        return httpx.Response(200, content=b"line one\nline two\n")

    respx.get(api_url).mock(
        return_value=httpx.Response(
            302,
            headers={
                "Location": blob_url,
                "X-RateLimit-Remaining": "41",
            },
        )
    )
    respx.get(blob_url).mock(side_effect=blob_response)

    download = client.download_job_log(
        "o",
        "r",
        101,
        keep_bytes=1024,
        ceiling_bytes=2048,
    )

    assert download.content == b"line one\nline two\n"
    assert download.original_bytes_seen == len(b"line one\nline two\n")
    assert download.kept_bytes == len(download.content)
    assert download.truncated is False
    assert seen_auth == [None]
    assert client.rate_limit_remaining == 41


@respx.mock
def test_log_tail_starts_at_line_boundary_and_keeps_final_error(
    client: GitHubClient,
) -> None:
    content = b"x" * 500 + b"\nERROR: final failure\n"
    api_url = "https://api.github.com/repos/o/r/actions/jobs/101/logs"
    respx.get(api_url).mock(return_value=httpx.Response(200, content=content))

    download = client.download_job_log(
        "o",
        "r",
        101,
        keep_bytes=40,
        ceiling_bytes=1000,
    )

    assert download.truncated is True
    assert download.original_bytes_seen == len(content)
    assert download.content.startswith(b"ERROR: final failure")
    assert download.content == b"ERROR: final failure\n"
    assert download.kept_bytes == len(download.content)
    assert download.kept_bytes <= 40


@respx.mock
def test_log_ceiling_marks_original_size(client: GitHubClient) -> None:
    api_url = "https://api.github.com/repos/o/r/actions/jobs/101/logs"
    respx.get(api_url).mock(return_value=httpx.Response(200, content=b"y" * 100))

    download = client.download_job_log(
        "o",
        "r",
        101,
        keep_bytes=10,
        ceiling_bytes=20,
    )

    assert download.truncated is True
    assert download.original_bytes_seen == "50MiB+"
    assert download.kept_bytes <= 10


@respx.mock
def test_has_success_run_for_same_workflow_and_sha(
    client: GitHubClient,
) -> None:
    run = WorkflowRun.model_validate(run_payload(run_id=1, head_sha="sha-1"))
    url = "https://api.github.com/repos/o/r/actions/workflows/10/runs"
    respx.get(url).mock(
        return_value=httpx.Response(
            200,
            json=runs_response(
                run_payload(
                    run_id=2,
                    head_sha="sha-1",
                    conclusion="success",
                )
            ),
        )
    )

    assert client.has_success_run("o", "r", run) is True


@respx.mock
def test_has_success_run_is_false_when_no_other_success(
    client: GitHubClient,
) -> None:
    run = WorkflowRun.model_validate(run_payload(run_id=1, head_sha="sha-1"))
    url = "https://api.github.com/repos/o/r/actions/workflows/10/runs"
    respx.get(url).mock(return_value=httpx.Response(200, json=runs_response()))

    assert client.has_success_run("o", "r", run) is False


@respx.mock
def test_has_success_run_falls_back_to_generic_runs(
    client: GitHubClient,
) -> None:
    payload = run_payload(run_id=1, head_sha="sha-1")
    payload["workflow_id"] = None
    run = WorkflowRun.model_validate(payload)
    respx.get(RUNS_URL).mock(
        return_value=httpx.Response(
            200,
            json=runs_response(
                run_payload(
                    run_id=2,
                    head_sha="sha-1",
                    conclusion="success",
                )
            ),
        )
    )

    assert client.has_success_run("o", "r", run) is True


def test_client_exposes_no_write_methods(client: GitHubClient) -> None:
    assert not hasattr(client, "post")
    assert not hasattr(client, "put")
    assert not hasattr(client, "patch")
    assert not hasattr(client, "delete")
