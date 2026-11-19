from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

type OriginalBytes = int | Literal["50MiB+"]


class Step(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str | None = None
    conclusion: str | None = None


class Job(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    name: str
    conclusion: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    steps: list[Step] = Field(default_factory=list)


class GitRef(BaseModel):
    model_config = ConfigDict(extra="ignore")

    sha: str | None = None
    ref: str | None = None


class PullRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    number: int | None = None
    base: GitRef | None = None


class HeadRepository(BaseModel):
    model_config = ConfigDict(extra="ignore")

    full_name: str | None = None


class WorkflowRun(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    name: str | None = None
    workflow_id: int | None = None
    workflow_path: str | None = None
    event: str | None = None
    conclusion: str | None = None
    head_sha: str | None = None
    head_branch: str | None = None
    run_attempt: int = 1
    run_started_at: datetime | None = None
    created_at: datetime
    html_url: str
    pull_requests: list[PullRequest] = Field(default_factory=list)
    head_repository: HeadRepository | None = None


class FailedJobMeta(BaseModel):
    id: int
    name: str
    started_at: datetime | None = None
    completed_at: datetime | None = None
    failed_step_names: list[str] = Field(default_factory=list)
    likely_aggregator: bool = False


class PullRequestMeta(BaseModel):
    number: int | None = None
    base_sha: str | None = None
    base_ref: str | None = None


class HeadRepositoryMeta(BaseModel):
    full_name: str | None = None


class LogMeta(BaseModel):
    job_id: int
    name: str
    file: str
    original_bytes_seen: OriginalBytes
    kept_bytes: int
    written_bytes: int
    truncated: bool
    truncation_side: Literal["head_dropped"] | None = None


class RunMeta(BaseModel):
    run_id: int
    repo: str
    workflow_name: str
    workflow_id: int | None
    workflow_path: str | None
    event: str | None
    conclusion: str | None
    head_sha: str | None
    head_branch: str | None
    run_attempt: int
    run_started_at: datetime | None
    created_at: datetime
    html_url: str
    pull_requests: list[PullRequestMeta] = Field(default_factory=list)
    head_repository: HeadRepositoryMeta | None = None
    failed_jobs: list[FailedJobMeta] = Field(default_factory=list)
    logs: list[LogMeta] = Field(default_factory=list)
    truncated_job_ids: list[int] = Field(default_factory=list)
    collected_at: datetime
    later_green_same_sha: bool | None = None


class RunsPage(BaseModel):
    total_count: int = 0
    workflow_runs: list[WorkflowRun] = Field(default_factory=list)


class JobsPage(BaseModel):
    total_count: int = 0
    jobs: list[Job] = Field(default_factory=list)
