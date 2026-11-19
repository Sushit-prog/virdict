from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from citriage.baselines.b0 import predict
from citriage.collect import DEFAULT_EXCLUDE_PATTERN, NON_CI_STEP_RE
from citriage.labels import LabelName
from citriage.models import FailedJobMeta
from citriage.parse.logparse import Evidence, ParsedLog, parse_log
from citriage.signature import signature_from_log
from citriage.storage import read_run_meta

_WORKFLOW_EXCLUDE_RE = re.compile(DEFAULT_EXCLUDE_PATTERN, re.IGNORECASE)

SEED = "citriage-cases-v1"
CASES_PATH = Path("data/cases.jsonl")


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    repo: str
    run_id: int
    run_dir_name: str = ""
    head_sha: str
    created_at: datetime
    workflow_name: str
    html_url: str = ""
    step: str
    failed_step_names: list[str] = Field(default_factory=list)
    job_id: int
    job_name: str
    sibling_job_ids: list[int]
    signature: str | None
    signature_rank: int | None
    likely_aggregator: bool
    flake_confirmed: bool
    in_benchmark: bool = True
    exclusion_reason: Literal["aggregator", "non_ci", "none"] = "none"
    split: Literal["dev", "test"]
    seed: str
    region_start: int
    region_end: int
    tools_detected: list[str]
    has_nontrivial_error: bool
    evidence: list[Evidence]
    b0_label: LabelName
    b0_culprit_file: str | None
    b0_evidence_ids: list[str]


def case_split(seed: str, repo: str, head_sha: str) -> Literal["dev", "test"]:
    digest = hashlib.sha1(f"{seed}:{repo}:{head_sha}".encode()).hexdigest()
    return "dev" if int(digest[:8], 16) % 100 < 35 else "test"


def _empty_parsed() -> ParsedLog:
    return ParsedLog(
        region_start=1,
        region_end=0,
        evidence=[],
        tools_detected=[],
        has_nontrivial_error=False,
    )


def _distinct_steps(failed_jobs: list[FailedJobMeta]) -> list[str]:
    steps: list[str] = []
    for job in failed_jobs:
        for step in job.failed_step_names:
            if step not in steps:
                steps.append(step)
    return steps


def _case_id(run_dir_name: str, step: str) -> str:
    digest = hashlib.sha1(step.encode("utf-8")).hexdigest()[:8]
    return f"{run_dir_name}:{digest}"


def _exclusion_reason(
    meta_workflow_name: str,
    meta_workflow_path: str | None,
    failed_step_names: list[str],
    likely_aggregator: bool,
) -> Literal["aggregator", "non_ci", "none"]:
    workflow = " ".join(part for part in (meta_workflow_name, meta_workflow_path or "") if part)
    if _WORKFLOW_EXCLUDE_RE.search(workflow):
        return "non_ci"
    if failed_step_names and all(NON_CI_STEP_RE.search(step) for step in failed_step_names):
        return "non_ci"
    if likely_aggregator:
        return "aggregator"
    return "none"


def build_cases(raw_dir: Path = Path("data/raw"), out: Path = CASES_PATH) -> list[Case]:
    cases: list[Case] = []
    for repo_dir in sorted(path for path in raw_dir.iterdir() if path.is_dir()):
        for run_path in sorted(path for path in repo_dir.iterdir() if path.is_dir()):
            meta = read_run_meta(run_path)
            if meta is None:
                continue
            log_paths = {
                log.job_id: run_path / log.file
                for log in meta.logs
                if (run_path / log.file).exists()
            }
            parsed_by_job: dict[int, ParsedLog] = {}
            head_sha = meta.head_sha or f"no-sha-{meta.run_id}"
            split = case_split(SEED, meta.repo, head_sha)
            run_dir_name = run_path.name
            failed_step_names = _distinct_steps(meta.failed_jobs)

            for step in failed_step_names:
                jobs = [job for job in meta.failed_jobs if step in job.failed_step_names]
                if not jobs:
                    continue
                representative = jobs[0]
                parsed = parsed_by_job.get(representative.id)
                if parsed is None:
                    log_path = log_paths.get(representative.id)
                    parsed = (
                        parse_log(log_path.read_text(encoding="utf-8", errors="replace"))
                        if log_path
                        else _empty_parsed()
                    )
                    parsed_by_job[representative.id] = parsed
                baseline = predict(parsed)
                reason = _exclusion_reason(
                    meta.workflow_name,
                    meta.workflow_path,
                    failed_step_names,
                    representative.likely_aggregator,
                )
                cases.append(
                    Case(
                        case_id=_case_id(run_dir_name, step),
                        repo=meta.repo,
                        run_id=meta.run_id,
                        run_dir_name=run_dir_name,
                        head_sha=head_sha,
                        created_at=meta.created_at,
                        workflow_name=meta.workflow_name,
                        html_url=meta.html_url,
                        step=step,
                        failed_step_names=failed_step_names,
                        job_id=representative.id,
                        job_name=representative.name,
                        sibling_job_ids=[job.id for job in jobs if job.id != representative.id],
                        signature=signature_from_log(parsed),
                        signature_rank=None,
                        likely_aggregator=representative.likely_aggregator,
                        flake_confirmed=meta.flake_confirmed,
                        in_benchmark=reason == "none",
                        exclusion_reason=reason,
                        split=split,
                        seed=SEED,
                        region_start=parsed.region_start,
                        region_end=parsed.region_end,
                        tools_detected=parsed.tools_detected,
                        has_nontrivial_error=parsed.has_nontrivial_error,
                        evidence=parsed.evidence,
                        b0_label=baseline.label,
                        b0_culprit_file=baseline.culprit_file,
                        b0_evidence_ids=baseline.evidence_ids,
                    )
                )

    by_signature: dict[str | None, list[Case]] = defaultdict(list)
    for case in cases:
        by_signature[case.signature].append(case)
    for group in by_signature.values():
        group.sort(key=lambda case: (case.created_at, case.repo, case.run_id, case.step))
        for rank, case in enumerate(group, start=1):
            case.signature_rank = rank

    cases.sort(key=lambda case: (case.repo, case.created_at, case.run_id, case.step))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for case in cases:
            handle.write(case.model_dump_json() + "\n")

    _print_counts(cases)
    return cases


def _print_counts(cases: list[Case]) -> None:
    print(f"seed: {SEED}")
    grouped = Counter((case.repo, case.split, case.flake_confirmed) for case in cases)
    for (repo, split, flake), count in sorted(grouped.items()):
        print(f"  {repo:28s} split={split:4s} flake={str(flake):5s} cases={count:4d}")
    print("totals by split:", dict(sorted(Counter(case.split for case in cases).items())))
    print(
        "totals by flake_confirmed:",
        dict(sorted(Counter(case.flake_confirmed for case in cases).items())),
    )
    print(
        "totals by in_benchmark:",
        dict(sorted(Counter(case.in_benchmark for case in cases).items())),
    )
    print(
        "totals by exclusion_reason:",
        dict(sorted(Counter(case.exclusion_reason for case in cases).items())),
    )
    signatures: dict[str, set[str]] = defaultdict(set)
    for case in cases:
        if case.signature is not None:
            signatures[case.split].add(case.signature)
    overlap = signatures["dev"] & signatures["test"]
    print(f"signatures in both dev and test: {len(overlap)}")
    print(f"cases written: {len(cases)}")


def load_cases(path: Path = CASES_PATH) -> list[Case]:
    if not path.exists():
        return []
    result: list[Case] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                result.append(Case.model_validate(json.loads(line)))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid case: {exc}") from exc
    return result


__all__ = ["CASES_PATH", "SEED", "Case", "build_cases", "case_split", "load_cases"]
