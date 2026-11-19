from __future__ import annotations

import re
from collections import deque
from collections.abc import Sequence
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

DEFAULT_EXCLUDE_PATTERN = (
    r"codeql|labeler|stale|triage|dependabot|release|publish|deploy|scorecard|"
    r"greet|assign|autolabel|codex|claude|copilot|classify"
)
AGGREGATOR_STATUS_RE = re.compile(r"^\S.*: (success|failure|skipped|cancelled)$")
AGGREGATOR_STEP_RE = re.compile(
    r"require|all.*(succeed|pass)|gate|status check",
    re.IGNORECASE,
)
PROCESS_EXIT_PREFIX = "Process completed with exit code"
MAX_CANDIDATES_PER_REPO = 40


@dataclass
class RepoSummary:
    repo: str
    runs_saved: int = 0
    jobs_saved: int = 0
    runs_skipped: int = 0
    candidates_evaluated: int = 0
    excluded: int = 0
    skipped_aggregator_only: int = 0
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

    @property
    def total_candidates_evaluated(self) -> int:
        return sum(repo.candidates_evaluated for repo in self.repos)

    @property
    def total_excluded(self) -> int:
        return sum(repo.excluded for repo in self.repos)

    @property
    def total_skipped_aggregator_only(self) -> int:
        return sum(repo.skipped_aggregator_only for repo in self.repos)


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


def _compile_excludes(
    patterns: Sequence[str],
    *,
    include_default_excludes: bool,
) -> list[re.Pattern[str]]:
    raw = [DEFAULT_EXCLUDE_PATTERN] if include_default_excludes else []
    raw.extend(patterns)
    return [re.compile(pattern, re.IGNORECASE) for pattern in raw]


def _is_excluded(run: WorkflowRun, patterns: list[re.Pattern[str]]) -> bool:
    values = [run.name, run.workflow_path]
    return any(
        pattern.search(value) for pattern in patterns for value in values if value is not None
    )


def _is_likely_aggregator(job: FailedJobMeta, log_text: str) -> bool:
    if len(job.failed_step_names) == 1 and AGGREGATOR_STEP_RE.search(job.failed_step_names[0]):
        return True

    lines = log_text.splitlines()[-40:]
    status_lines = sum(1 for line in lines if AGGREGATOR_STATUS_RE.match(line))
    if status_lines < 3:
        return False

    for line in lines:
        if "##[error]" not in line:
            continue
        message = line.split("##[error]", 1)[1].strip()
        if not message.startswith(PROCESS_EXIT_PREFIX):
            return False
    return True


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
) -> tuple[list[LogMeta], dict[str, bytes], set[int]]:
    logs: list[LogMeta] = []
    files: dict[str, bytes] = {}
    aggregator_ids: set[int] = set()
    for job in failed_jobs:
        download = client.download_job_log(
            owner,
            repo,
            job.id,
            keep_bytes=DEFAULT_KEEP_BYTES,
            ceiling_bytes=DEFAULT_CEILING_BYTES,
        )
        text = download.content.decode("utf-8", errors="replace")
        if _is_likely_aggregator(job, text):
            aggregator_ids.add(job.id)
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
    return logs, files, aggregator_ids


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
    exclude_patterns: list[re.Pattern[str]],
) -> None:
    owner, name = repo_summary.repo.split("/", 1)
    failed_runs = client.list_failed_runs(
        owner,
        name,
        since=now - timedelta(days=80),
    )

    kept_runs: list[WorkflowRun] = []
    for run in failed_runs:
        if _is_excluded(run, exclude_patterns):
            repo_summary.excluded += 1
        else:
            kept_runs.append(run)

    groups: dict[str, deque[WorkflowRun]] = {}
    for run in kept_runs:
        key = str(run.workflow_id) if run.workflow_id is not None else (run.name or "unknown")
        groups.setdefault(key, deque()).append(run)

    used_signatures: dict[str, set[tuple[str, ...]]] = {key: set() for key in groups}

    while (
        repo_summary.runs_saved < per_repo
        and repo_summary.candidates_evaluated < MAX_CANDIDATES_PER_REPO
        and any(groups.values())
    ):
        progressed = False
        for key in list(groups):
            if (
                repo_summary.runs_saved >= per_repo
                or repo_summary.candidates_evaluated >= MAX_CANDIDATES_PER_REPO
            ):
                break
            queue = groups[key]
            while queue:
                if (
                    repo_summary.runs_saved >= per_repo
                    or repo_summary.candidates_evaluated >= MAX_CANDIDATES_PER_REPO
                ):
                    break
                run = queue.popleft()
                repo_summary.candidates_evaluated += 1
                path = run_dir(out, repo_summary.repo, run.id)
                existing_meta = read_run_meta(path) if path.exists() else None

                if path.exists() and existing_meta is None:
                    repo_summary.errors.append(
                        f"run {run.id}: existing run directory has no readable meta.json; skipped"
                    )
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
                    repo_summary.runs_skipped += 1
                    progressed = True
                    break

                try:
                    logs, log_files, aggregator_ids = _fetch_logs(
                        client,
                        owner,
                        name,
                        failed_jobs,
                        extra_secrets,
                    )
                    failed_jobs = [
                        job.model_copy(update={"likely_aggregator": job.id in aggregator_ids})
                        for job in failed_jobs
                    ]
                    if all(job.likely_aggregator for job in failed_jobs):
                        repo_summary.skipped_aggregator_only += 1
                        progressed = True
                        break

                    later_green = _later_green(client, owner, name, run)
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
    exclude_patterns: Sequence[str] = (),
    include_default_excludes: bool = True,
    now: datetime | None = None,
) -> CollectSummary:
    if per_repo < 1:
        raise ValueError("per_repo must be positive")

    compiled_excludes = _compile_excludes(
        exclude_patterns,
        include_default_excludes=include_default_excludes,
    )
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
                exclude_patterns=compiled_excludes,
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
