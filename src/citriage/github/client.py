from __future__ import annotations

import contextlib
import random
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from types import TracebackType
from typing import Any, Literal, Self, cast

import httpx
from pydantic import ValidationError

from citriage.errors import (
    AuthenticationError,
    GitHubAPIError,
    LogUnavailableError,
    MissingTokenError,
    NetworkError,
    RateLimitExhausted,
)
from citriage.models import (
    Job,
    JobsPage,
    OriginalBytes,
    RunsPage,
    WorkflowRun,
)

DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)
DEFAULT_KEEP_BYTES = 5 * 1024 * 1024
DEFAULT_CEILING_BYTES = 50 * 1024 * 1024


@dataclass(frozen=True)
class LogDownload:
    content: bytes
    original_bytes_seen: OriginalBytes
    kept_bytes: int
    truncated: bool


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _parse_epoch(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(int(value), UTC)
    except (ValueError, OSError):
        return None


def _parse_retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        pass
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return (parsed - datetime.now(UTC)).total_seconds()


class GitHubClient:
    def __init__(
        self,
        token: str,
        *,
        timeout: httpx.Timeout | float = DEFAULT_TIMEOUT,
        sleep: Callable[[float], None] = time.sleep,
        max_sleep_seconds: float = 60.0,
        retry_attempts: int = 3,
        api_url: str = "https://api.github.com",
    ) -> None:
        if not token:
            raise MissingTokenError("GITHUB_TOKEN is not set.")
        self._token = token
        self._sleep = sleep
        self._max_sleep_seconds = max_sleep_seconds
        self._retry_attempts = max(1, retry_attempts)
        self._client = httpx.Client(
            base_url=api_url,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "citriage",
            },
            timeout=timeout,
            follow_redirects=False,
        )
        self.rate_limit_remaining: int | None = None
        self.rate_limit_reset: datetime | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def _record_rate_limit(self, response: httpx.Response) -> None:
        for item in (*response.history, response):
            remaining = item.headers.get("x-ratelimit-remaining")
            if remaining is not None:
                with contextlib.suppress(ValueError):
                    self.rate_limit_remaining = int(remaining)
            reset = _parse_epoch(item.headers.get("x-ratelimit-reset"))
            if reset is not None:
                self.rate_limit_reset = reset

    def _maybe_wait_for_rate_limit(self) -> None:
        if self.rate_limit_remaining != 0:
            return
        reset = self.rate_limit_reset
        if reset is None:
            raise RateLimitExhausted(None)
        wait = (reset - datetime.now(UTC)).total_seconds()
        if wait <= 0:
            self.rate_limit_remaining = None
            return
        if wait <= self._max_sleep_seconds:
            self._sleep(wait + 1.0)
            self.rate_limit_remaining = None
            return
        raise RateLimitExhausted(reset)

    def _is_rate_limited(self, response: httpx.Response) -> bool:
        if response.status_code not in (403, 429):
            return False
        if response.headers.get("retry-after") is not None:
            return True
        remaining = response.headers.get("x-ratelimit-remaining")
        return remaining is not None and bool(remaining == "0")

    def _wait_for_reset(self, response: httpx.Response) -> None:
        now = datetime.now(UTC)
        retry_after = _parse_retry_after(response.headers.get("retry-after"))
        if retry_after is not None:
            wait = retry_after
            reset_at = now + timedelta(seconds=max(0.0, wait))
        elif self.rate_limit_reset is not None:
            reset_at = self.rate_limit_reset
            wait = (reset_at - now).total_seconds()
        else:
            raise RateLimitExhausted(None)

        if wait <= 0:
            self.rate_limit_remaining = None
            return
        if wait <= self._max_sleep_seconds:
            self._sleep(wait + 1.0)
            self.rate_limit_remaining = None
            return
        raise RateLimitExhausted(reset_at)

    def _backoff(self, attempt: int) -> float:
        return float(min(8.0, 0.5 * (2**attempt)) + random.uniform(0, 0.25))

    def _next_url(self, response: httpx.Response) -> str | None:
        link = response.links.get("next")
        if link is None:
            return None
        raw = link.get("url")
        if not raw:
            return None
        url = response.request.url.join(raw)
        if url.host != self._client.base_url.host:
            raise GitHubAPIError(
                "Refusing to follow a pagination link to a different host.",
                url=str(url),
            )
        return str(url)

    def _request(
        self,
        url: str,
        *,
        params: dict[str, str | int] | None = None,
        follow_redirects: bool = False,
    ) -> httpx.Response:
        last_status: int | None = None
        last_url: str | None = None
        for attempt in range(self._retry_attempts):
            self._maybe_wait_for_rate_limit()
            try:
                response = self._client.get(
                    url,
                    params=params,
                    follow_redirects=follow_redirects,
                )
            except httpx.TimeoutException as exc:
                raise NetworkError(f"Timed out requesting {url}.") from exc
            except httpx.HTTPError as exc:
                raise NetworkError(f"Request failed for {url}.") from exc

            self._record_rate_limit(response)

            if response.status_code == 401:
                raise AuthenticationError(
                    "GitHub rejected GITHUB_TOKEN (401).",
                    status_code=401,
                    url=url,
                )
            if self._is_rate_limited(response):
                last_status = response.status_code
                last_url = url
                self._wait_for_reset(response)
                continue
            if 500 <= response.status_code < 600:
                last_status = response.status_code
                last_url = url
                if attempt + 1 < self._retry_attempts:
                    self._sleep(self._backoff(attempt))
                    continue
                raise GitHubAPIError(
                    f"GitHub returned {response.status_code} for {url}.",
                    status_code=response.status_code,
                    url=url,
                )
            if response.status_code >= 400:
                raise GitHubAPIError(
                    f"GitHub returned {response.status_code} for {url}.",
                    status_code=response.status_code,
                    url=url,
                )
            return response
        if last_status in (403, 429):
            raise RateLimitExhausted(self.rate_limit_reset)
        raise GitHubAPIError(
            f"Retries exhausted for {url}.",
            status_code=last_status,
            url=last_url,
        )

    def _parse_runs_page(self, payload: dict[str, Any]) -> RunsPage:
        try:
            return RunsPage.model_validate(payload)
        except ValidationError as exc:
            raise GitHubAPIError("GitHub returned an invalid workflow runs payload.") from exc

    def _parse_jobs_page(self, payload: dict[str, Any]) -> JobsPage:
        try:
            return JobsPage.model_validate(payload)
        except ValidationError as exc:
            raise GitHubAPIError("GitHub returned an invalid workflow jobs payload.") from exc

    def _iter_pages(
        self,
        path: str,
        params: dict[str, str | int],
    ) -> Iterator[dict[str, Any]]:
        url: str | None = path
        current_params: dict[str, str | int] | None = params
        while url is not None:
            response = self._request(url, params=current_params)
            try:
                payload = response.json()
            except ValueError as exc:
                raise GitHubAPIError(
                    f"GitHub returned invalid JSON for {url}.",
                    url=url,
                ) from exc
            if not isinstance(payload, dict):
                raise GitHubAPIError(
                    f"GitHub returned an unexpected JSON payload for {url}.",
                    url=url,
                )
            yield cast(dict[str, Any], payload)
            url = self._next_url(response)
            current_params = None

    def _list_runs(
        self,
        owner: str,
        repo: str,
        *,
        status: Literal["failure", "success"],
        since: datetime,
        per_page: int,
    ) -> list[WorkflowRun]:
        cutoff = _as_utc(since)
        path = f"/repos/{owner}/{repo}/actions/runs"
        params: dict[str, str | int] = {
            "status": status,
            "per_page": per_page,
        }
        runs: list[WorkflowRun] = []
        for payload in self._iter_pages(path, params):
            page = self._parse_runs_page(payload)
            if not page.workflow_runs:
                break
            for run in page.workflow_runs:
                if _as_utc(run.created_at) >= cutoff:
                    runs.append(run)
            if all(_as_utc(run.created_at) < cutoff for run in page.workflow_runs):
                break
        return runs

    def list_failed_runs(
        self,
        owner: str,
        repo: str,
        *,
        since: datetime,
        per_page: int = 50,
    ) -> list[WorkflowRun]:
        return self._list_runs(
            owner,
            repo,
            status="failure",
            since=since,
            per_page=per_page,
        )

    def list_success_runs(
        self,
        owner: str,
        repo: str,
        *,
        since: datetime,
        per_page: int = 50,
    ) -> list[WorkflowRun]:
        return self._list_runs(
            owner,
            repo,
            status="success",
            since=since,
            per_page=per_page,
        )

    def _list_failed_jobs(self, path: str, *, per_page: int) -> list[Job]:
        params: dict[str, str | int] = {"per_page": per_page}
        jobs: list[Job] = []
        for payload in self._iter_pages(path, params):
            page = self._parse_jobs_page(payload)
            jobs.extend(job for job in page.jobs if job.conclusion == "failure")
        return jobs

    def list_failed_jobs(
        self,
        owner: str,
        repo: str,
        run_id: int,
        *,
        per_page: int = 100,
    ) -> list[Job]:
        path = f"/repos/{owner}/{repo}/actions/runs/{run_id}/jobs"
        return self._list_failed_jobs(path, per_page=per_page)

    def list_failed_jobs_for_attempt(
        self,
        owner: str,
        repo: str,
        run_id: int,
        attempt: int,
        *,
        per_page: int = 100,
    ) -> list[Job]:
        if attempt < 1:
            raise ValueError("attempt must be positive")
        path = f"/repos/{owner}/{repo}/actions/runs/{run_id}/attempts/{attempt}/jobs"
        return self._list_failed_jobs(path, per_page=per_page)

    def _read_tail(
        self,
        response: httpx.Response,
        keep_bytes: int,
        ceiling_bytes: int,
    ) -> LogDownload:
        tail = bytearray()
        total = 0
        ceiling_exceeded = False

        for chunk in response.iter_bytes(chunk_size=64 * 1024):
            total += len(chunk)
            if total > ceiling_bytes:
                allowed = max(0, ceiling_bytes - (total - len(chunk)))
                tail.extend(chunk[:allowed])
                if len(tail) > keep_bytes:
                    del tail[: len(tail) - keep_bytes]
                ceiling_exceeded = True
                break
            tail.extend(chunk)
            if len(tail) > keep_bytes:
                del tail[: len(tail) - keep_bytes]

        original: OriginalBytes = "50MiB+" if ceiling_exceeded else total
        truncated = ceiling_exceeded or total > keep_bytes
        content = bytes(tail)
        if truncated:
            first_newline = content.find(b"\n")
            content = b"" if first_newline < 0 else content[first_newline + 1 :]

        return LogDownload(
            content=content,
            original_bytes_seen=original,
            kept_bytes=len(content),
            truncated=truncated,
        )

    def download_job_log(
        self,
        owner: str,
        repo: str,
        job_id: int,
        *,
        keep_bytes: int = DEFAULT_KEEP_BYTES,
        ceiling_bytes: int = DEFAULT_CEILING_BYTES,
    ) -> LogDownload:
        if keep_bytes <= 0:
            raise ValueError("keep_bytes must be positive")
        if ceiling_bytes < keep_bytes:
            raise ValueError("ceiling_bytes must be at least keep_bytes")

        path = f"/repos/{owner}/{repo}/actions/jobs/{job_id}/logs"
        last_status: int | None = None

        for attempt in range(self._retry_attempts):
            self._maybe_wait_for_rate_limit()
            retry_delay: float | None = None
            try:
                with self._client.stream(
                    "GET",
                    path,
                    follow_redirects=True,
                ) as response:
                    self._record_rate_limit(response)
                    last_status = response.status_code

                    if response.status_code == 401:
                        raise AuthenticationError(
                            "GitHub rejected GITHUB_TOKEN (401).",
                            status_code=401,
                            url=path,
                        )
                    if self._is_rate_limited(response):
                        self._wait_for_reset(response)
                        continue
                    if 500 <= response.status_code < 600:
                        if attempt + 1 < self._retry_attempts:
                            retry_delay = self._backoff(attempt)
                        else:
                            raise GitHubAPIError(
                                f"GitHub returned {response.status_code} for {path}.",
                                status_code=response.status_code,
                                url=path,
                            )
                    elif response.status_code >= 400:
                        raise LogUnavailableError(
                            f"Job log {job_id} unavailable ({response.status_code}).",
                            status_code=response.status_code,
                            url=path,
                        )
                    else:
                        return self._read_tail(response, keep_bytes, ceiling_bytes)
            except httpx.TimeoutException as exc:
                raise NetworkError(f"Timed out downloading job log {job_id}.") from exc
            except httpx.HTTPError as exc:
                raise NetworkError(f"Download failed for job log {job_id}.") from exc

            if retry_delay is not None:
                self._sleep(retry_delay)

        if last_status in (403, 429):
            raise RateLimitExhausted(self.rate_limit_reset)
        raise GitHubAPIError(
            f"Retries exhausted downloading job log {job_id}.",
            status_code=last_status,
            url=path,
        )

    def has_success_run(
        self,
        owner: str,
        repo: str,
        run: WorkflowRun,
    ) -> bool | None:
        sha = run.head_sha
        if not sha:
            return None

        if run.workflow_id is not None:
            path = f"/repos/{owner}/{repo}/actions/workflows/{run.workflow_id}/runs"
            params: dict[str, str | int] = {
                "head_sha": sha,
                "status": "success",
                "per_page": 100,
            }
            for payload in self._iter_pages(path, params):
                page = self._parse_runs_page(payload)
                if any(
                    candidate.id != run.id and candidate.conclusion == "success"
                    for candidate in page.workflow_runs
                ):
                    return True
            return False

        if not run.name:
            return None

        path = f"/repos/{owner}/{repo}/actions/runs"
        params = {
            "head_sha": sha,
            "status": "success",
            "per_page": 100,
        }
        for payload in self._iter_pages(path, params):
            page = self._parse_runs_page(payload)
            if any(
                candidate.id != run.id
                and candidate.conclusion == "success"
                and candidate.name == run.name
                for candidate in page.workflow_runs
            ):
                return True
        return False


__all__ = [
    "DEFAULT_CEILING_BYTES",
    "DEFAULT_KEEP_BYTES",
    "GitHubClient",
    "LogDownload",
]
