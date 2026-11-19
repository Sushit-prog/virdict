import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from citriage.cases import SEED, build_cases, case_split, load_cases
from citriage.labels import LabelName
from citriage.models import FailedJobMeta, LogMeta, RunMeta

LOG = (Path(__file__).parent / "fixtures" / "logs" / "pytest_failure.log").read_text(
    encoding="utf-8"
)


def _meta(run_id: int, job_ids: list[int], created_at: datetime, flake: bool) -> RunMeta:
    return RunMeta(
        run_id=run_id,
        repo="o/r",
        workflow_name="CI",
        workflow_id=10,
        workflow_path=None,
        event="push",
        conclusion="failure",
        head_sha="abc123",
        head_branch="main",
        run_attempt=1,
        run_started_at=created_at,
        created_at=created_at,
        html_url=f"https://github.com/o/r/actions/runs/{run_id}",
        failed_jobs=[
            FailedJobMeta(
                id=job_id,
                name=f"job {job_id}",
                failed_step_names=["pytest"],
                likely_aggregator=False,
            )
            for job_id in job_ids
        ],
        logs=[
            LogMeta(
                job_id=job_id,
                name=f"job {job_id}",
                file=f"job_{job_id}.log",
                original_bytes_seen=len(LOG),
                kept_bytes=len(LOG),
                written_bytes=len(LOG),
                truncated=False,
            )
            for job_id in job_ids
        ],
        collected_at=created_at,
        later_green_same_sha=None,
        flake_confirmed=flake,
    )


def test_build_cases_indexes_steps_siblings_signatures_and_splits(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    base = datetime(2026, 11, 20, 12, 0, tzinfo=UTC)
    for run_id, job_ids, flake in ((1, [10, 11], False), (2, [20], True)):
        run_path = raw / "o__r" / str(run_id)
        run_path.mkdir(parents=True)
        meta = _meta(run_id, job_ids, base + timedelta(hours=run_id), flake)
        (run_path / "meta.json").write_text(meta.model_dump_json(), encoding="utf-8")
        for job_id in job_ids:
            (run_path / f"job_{job_id}.log").write_text(LOG, encoding="utf-8")

    out = tmp_path / "cases.jsonl"
    cases = build_cases(raw, out)
    loaded = load_cases(out)

    assert len(cases) == 2 == len(loaded)
    assert all(case.step == "pytest" for case in cases)
    assert next(case for case in cases if case.job_id == 10).sibling_job_ids == [11]
    assert next(case for case in cases if case.job_id == 20).sibling_job_ids == []
    assert sorted(case.signature_rank for case in cases) == [1, 2]
    assert len({case.signature for case in cases}) == 1
    assert all(case.seed == SEED for case in cases)
    assert all(case.split == case_split(SEED, "o/r", "abc123") for case in cases)
    assert all(case.b0_label == LabelName.TEST_FAILURE for case in cases)
    assert sum(case.flake_confirmed for case in cases) == 1
    assert all(case.in_benchmark for case in cases)
    assert all(case.exclusion_reason == "none" for case in cases)
    assert all(case.html_url.startswith("https://github.com/o/r/actions/runs/") for case in cases)


def test_case_id_uses_run_directory_and_step(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    run_path = raw / "o__r" / "2_a2"
    run_path.mkdir(parents=True)
    base = datetime(2026, 11, 20, 12, 0, tzinfo=UTC)
    meta = _meta(2, [20], base, True)
    meta.failed_jobs[0].failed_step_names = ["Run tests"]
    (run_path / "meta.json").write_text(meta.model_dump_json(), encoding="utf-8")
    (run_path / "job_20.log").write_text(LOG, encoding="utf-8")

    case = build_cases(raw, tmp_path / "cases.jsonl")[0]

    digest = hashlib.sha1(b"Run tests").hexdigest()[:8]
    assert case.case_id == f"2_a2:{digest}"
    assert case.run_dir_name == "2_a2"
    assert case.failed_step_names == ["Run tests"]
    assert case.html_url == meta.html_url


@pytest.mark.parametrize(
    ("workflow_name", "steps", "aggregator", "reason"),
    [
        ("Backport label added", ["Run actions/github-script@v9.0.0"], False, "non_ci"),
        ("CI", ["Run actions/github-script@v9.0.0", "Run tests"], False, "none"),
        ("CI", ["Decide whether to run tests"], True, "aggregator"),
    ],
)
def test_case_exclusion_reasons(
    tmp_path: Path,
    workflow_name: str,
    steps: list[str],
    aggregator: bool,
    reason: str,
) -> None:
    raw = tmp_path / "raw"
    run_path = raw / "o__r" / "1"
    run_path.mkdir(parents=True)
    base = datetime(2026, 11, 20, 12, 0, tzinfo=UTC)
    meta = _meta(1, [10], base, False)
    meta.workflow_name = workflow_name
    meta.failed_jobs[0].failed_step_names = steps
    meta.failed_jobs[0].likely_aggregator = aggregator
    (run_path / "meta.json").write_text(meta.model_dump_json(), encoding="utf-8")
    (run_path / "job_10.log").write_text(LOG, encoding="utf-8")

    case = build_cases(raw, tmp_path / "cases.jsonl")[0]

    assert case.exclusion_reason == reason
    assert case.in_benchmark is (reason == "none")
