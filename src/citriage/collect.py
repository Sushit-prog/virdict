from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from citriage.errors import (
    AuthenticationError,
    CitriageError,
    RateLimitExhausted,
)
from citriage.github.client import (
    DEFAULT_CEILING_BYTES,
    DEFAULT_KEEP_BYTES,
    GitHubClient,
)
from citriage.models import (
    FailedJobMeta,
    HeadRepositoryMeta,
    Job,
    LogMeta,
    PullRequestMeta,
    RunMeta,
    WorkflowRun,
)
from citriage.redact import redact
from citriage.storage import read_run_meta, run_dir, write_run


@dataclass
class RepoSummary:
    repo: str
    runs_saved: int = 0
    jobs_saved: int = 0
    runs_skipped: int = 0
    log_sizes: list[int] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class CollectSummary:
    repos: list[RepoSummary] = field(default_factory=list)
    rate_limit_remaining: int | None = None
    stopped_reason: str | None = None

    @property
    def has_errors(self) -> bool:
        return any(repo.errors for repo in self.repos)

    @property
    def total_skipped(self) -> int:
        return sum(repo.runs_skipped for repo in self.repos)


def _job_meta(job: Job) -> FailedJobMeta:
    return FailedJobMeta(
        id=job.id,
        name=job.name,
        started_at=job.started_at,
        completed_at=job.completed_at,
        failed_step_names=[
            step.name
            for step in job.steps
            if step.conclusion == "failure" and step.name is not None
        ],
    )


def _signature(job: FailedJobMeta, extra_secrets: tuple[str, ...]) -> str:
    return redact(job.name, extra_secrets).strip().lower()


def _later_green(
    client: GitHubClient,
    owner: str,
    repo: str,
    run: WorkflowRun,
) -> bool | None:
    try:
        return client.has_success_run(owner, repo, run)
    except (AuthenticationError, RateLimitExhausted):
        raise
    except CitriageError:
        return None


def _fetch_logs(
    client: GitHubClient,
    owner: str,
    repo: str,
    failed_jobs: list[FailedJobMeta],
    extra_secrets: tuple[str, ...],
) -> tuple[list[LogMeta], dict[str, bytes]]:
    logs: list[LogMeta] = []
    files: dict[str, bytes] = {}
    for job in failed_jobs:
        download = client.download_job_log(
            owner,
            repo,
            job.id,
            keep_bytes=DEFAULT_KEEP_BYTES,
            ceiling_bytes=DEFAULT_CEILING_BYTES,
        )
        text = download.content.decode("utf-8", errors="replace")
        content = redact(text, extra_secrets).encode("utf-8")
        written_truncated = False
        if len(content) > DEFAULT_KEEP_BYTES:
            content = content[-DEFAULT_KEEP_BYTES:]
            first_newline = content.find(b"\n")
            content = b"" if first_newline < 0 else content[first_newline + 1 :]
            written_truncated = True
        truncated = download.truncated or written_truncated
        filename = f"job_{job.id}.log"
        files[filename] = content
        logs.append(
            LogMeta(
                job_id=job.id,
                name=job.name,
                file=filename,
                original_bytes_seen=download.original_bytes_seen,
                kept_bytes=download.kept_bytes,
                written_bytes=len(content),
                truncated=truncated,
                truncation_side="head_dropped" if truncated else None,
            )
        )
    return logs, files


def _build_meta(
    run: WorkflowRun,
    repo: str,
    failed_jobs: list[FailedJobMeta],
    logs: list[LogMeta],
    later_green_same_sha: bool | None,
    collected_at: datetime,
) -> RunMeta:
    pull_requests = [
        PullRequestMeta(
            number=request.number,
            base_sha=request.base.sha if request.base is not None else None,
            base_ref=request.base.ref if request.base is not None else None,
        )
        for request in run.pull_requests
    ]
    head_repository = (
        HeadRepositoryMeta(full_name=run.head_repository.full_name)
        if run.head_repository is not None
        else None
    )
    return RunMeta(
        run_id=run.id,
        repo=repo,
        workflow_name=run.name or "",
        workflow_id=run.workflow_id,
        workflow_path=run.workflow_path,
        event=run.event,
        conclusion=run.conclusion,
        head_sha=run.head_sha,
        head_branch=run.head_branch,
        run_attempt=run.run_attempt,
        run_started_at=run.run_started_at,
        created_at=run.created_at,
        html_url=run.html_url,
        pull_requests=pull_requests,
        head_repository=head_repository,
        failed_jobs=failed_jobs,
        logs=logs,
        truncated_job_ids=[log.job_id for log in logs if log.truncated],
        collected_at=collected_at,
        later_green_same_sha=later_green_same_sha,
    )


def _collect_repo(
    client: GitHubClient,
    repo_summary: RepoSummary,
    *,
    per_repo: int,
    out: Path,
    now: datetime,
    extra_secrets: tuple[str, ...],
) -> None:
    owner, name = repo_summary.repo.split("/", 1)
    failed_runs = client.list_failed_runs(
        owner,
        name,
        since=now - timedelta(days=80),
    )

    groups: dict[str, deque[WorkflowRun]] = {}
    for run in failed_runs:
        key = str(run.workflow_id) if run.workflow_id is not None else (run.name or "unknown")
        groups.setdefault(key, deque()).append(run)

    used_signatures: dict[str, set[tuple[str, ...]]] = {key: set() for key in groups}
    handled = 0

    while handled < per_repo and any(groups.values()):
        progressed = False
        for key in list(groups):
            queue = groups[key]
            while queue:
                run = queue.popleft()
                path = run_dir(out, repo_summary.repo, run.id)
                existing_meta = read_run_meta(path) if path.exists() else None

                if path.exists() and existing_meta is None:
                    repo_summary.errors.append(
                        f"run {run.id}: existing run directory has no readable meta.json; skipped"
                    )
                    handled += 1
                    repo_summary.runs_skipped += 1
                    progressed = True
                    break

                if existing_meta is not None:
                    failed_jobs = existing_meta.failed_jobs
                    already_saved = True
                else:
                    try:
                        jobs = client.list_failed_jobs(owner, name, run.id)
                    except (AuthenticationError, RateLimitExhausted):
                        raise
                    except CitriageError as exc:
                        repo_summary.errors.append(f"run {run.id}: {exc}")
                        continue
                    failed_jobs = [_job_meta(job) for job in jobs]
                    already_saved = False

                if not failed_jobs:
                    continue

                signature = tuple(sorted(_signature(job, extra_secrets) for job in failed_jobs))
                if signature in used_signatures[key]:
                    continue
                used_signatures[key].add(signature)

                if already_saved:
                    handled += 1
                    repo_summary.runs_skipped += 1
                    progressed = True
                    break

                try:
                    later_green = _later_green(client, owner, name, run)
                    logs, log_files = _fetch_logs(
                        client,
                        owner,
                        name,
                        failed_jobs,
                        extra_secrets,
                    )
                    meta = _build_meta(
                        run,
                        repo_summary.repo,
                        failed_jobs,
                        logs,
                        later_green,
                        now,
                    )
                    wrote = write_run(
                        path,
                        meta,
                        log_files,
                        extra_secrets=extra_secrets,
                    )
                except (AuthenticationError, RateLimitExhausted):
                    raise
                except CitriageError as exc:
                    repo_summary.errors.append(f"run {run.id}: {exc}")
                    continue

                handled += 1
                if wrote:
                    repo_summary.runs_saved += 1
                    repo_summary.jobs_saved += len(logs)
                    repo_summary.log_sizes.extend(log.written_bytes for log in logs)
                else:
                    repo_summary.runs_skipped += 1
                progressed = True
                break

        if not progressed:
            break


def collect_repos(
    client: GitHubClient,
    repos: list[str],
    *,
    per_repo: int,
    out: Path,
    extra_secrets: tuple[str, ...] = (),
    now: datetime | None = None,
) -> CollectSummary:
    if per_repo < 1:
        raise ValueError("per_repo must be positive")

    collected_at = now or datetime.now(UTC)
    summary = CollectSummary(
        repos=[RepoSummary(repo=repo) for repo in repos],
    )

    for repo_summary in summary.repos:
        try:
            _collect_repo(
                client,
                repo_summary,
                per_repo=per_repo,
                out=out,
                now=collected_at,
                extra_secrets=extra_secrets,
            )
        except RateLimitExhausted as exc:
            summary.stopped_reason = str(exc)
        except AuthenticationError:
            summary.rate_limit_remaining = client.rate_limit_remaining
            raise
        except CitriageError as exc:
            repo_summary.errors.append(str(exc))
        finally:
            summary.rate_limit_remaining = client.rate_limit_remaining

        if summary.stopped_reason is not None:
            break

    return summary
