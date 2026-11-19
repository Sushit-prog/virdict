from __future__ import annotations

import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from citriage.bench.metrics import (
    BENCH_SEED,
    BOOTSTRAP_RESAMPLES,
    AggregateMetrics,
    EvidenceCaseMetrics,
    aggregate_metrics,
    culprit_match,
    score_evidence,
)
from citriage.bench.systems import Prediction, System
from citriage.cases import CASES_PATH, Case, load_cases
from citriage.labels import LABELS_PATH, Label, LabelName, latest_labels, load_labels
from citriage.storage import log_path

FROZEN_PATH = Path("PARSER_FROZEN.md")
DEFAULT_RAW_ROOT = Path("data/raw")
Split = Literal["dev", "test"]


class BenchError(ValueError):
    pass


@dataclass(frozen=True)
class GoldSelection:
    pairs: tuple[tuple[Case, Label], ...]
    candidate_cases: int
    metadata_mismatches: int


@dataclass(frozen=True)
class CaseScore:
    case: Case
    gold: Label
    prediction: Prediction
    evidence: EvidenceCaseMetrics
    culprit_hit: bool | None


@dataclass(frozen=True)
class BenchResult:
    system: str
    split: Split
    labeler: str
    git_commit: str
    seed: int
    resamples: int
    date: str
    n_labeled: int
    candidate_cases: int
    skipped_no_label: int
    metadata_mismatches: int
    rows: tuple[CaseScore, ...]
    overall: AggregateMetrics
    per_repo: dict[str, AggregateMetrics]


def split_guard(split: Split, confirm_test: bool, frozen_path: Path = FROZEN_PATH) -> str | None:
    if split != "test":
        return None
    if not confirm_test:
        raise BenchError("--split test requires --confirm-test")
    if not frozen_path.exists():
        raise BenchError("--split test requires PARSER_FROZEN.md")
    return "WARNING: test labels must not drive parser or B0 changes."


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


def select_gold(
    cases: list[Case],
    labels: list[Label],
    split: Split,
    labeler: str,
) -> GoldSelection:
    if labeler.endswith("-r2"):
        raise BenchError("labelers ending in -r2 are excluded from benchmark gold")
    current = latest_labels(labels)
    pairs: list[tuple[Case, Label]] = []
    candidates = 0
    mismatches = 0
    for case in cases:
        if case.split != split or not case.in_benchmark:
            continue
        candidates += 1
        record = current.get((case.case_id, labeler))
        if record is None:
            continue
        if record.repo != case.repo or record.run_id != case.run_id or record.job_id != case.job_id:
            mismatches += 1
            continue
        pairs.append((case, record))
    return GoldSelection(
        pairs=tuple(pairs),
        candidate_cases=candidates,
        metadata_mismatches=mismatches,
    )


def _raw_lines(raw_root: Path, case: Case) -> list[str] | None:
    path = log_path(
        raw_root,
        case.repo,
        case.run_id,
        case.job_id,
        run_dir_name=case.run_dir_name or None,
    )
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def _aggregate_rows(
    rows: list[CaseScore],
    *,
    seed: int,
    resamples: int,
) -> AggregateMetrics:
    gold_labels: list[LabelName] = []
    for row in rows:
        assert row.gold.label is not None
        gold_labels.append(row.gold.label)
    return aggregate_metrics(
        gold_labels,
        [row.prediction.label for row in rows],
        [row.evidence for row in rows],
        [row.culprit_hit for row in rows],
        seed=seed,
        resamples=resamples,
    )


def run_bench(
    *,
    system: System,
    system_name: str,
    split: Split,
    labeler: str,
    cases_path: Path = CASES_PATH,
    labels_path: Path = LABELS_PATH,
    raw_root: Path = DEFAULT_RAW_ROOT,
    confirm_test: bool = False,
    frozen_path: Path = FROZEN_PATH,
    seed: int = BENCH_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> BenchResult:
    split_guard(split, confirm_test, frozen_path)
    cases = load_cases(cases_path)
    labels = load_labels(labels_path)
    selection = select_gold(cases, labels, split, labeler)
    if not selection.pairs:
        raise BenchError(
            f"no current gold labels for labeler {labeler!r} on split {split!r}; "
            "nothing to benchmark"
        )

    raw_cache: dict[Path, list[str] | None] = {}
    rows: list[CaseScore] = []
    for case, gold in selection.pairs:
        prediction = system.predict(case)
        raw_key = log_path(
            raw_root,
            case.repo,
            case.run_id,
            case.job_id,
            run_dir_name=case.run_dir_name or None,
        )
        if raw_key not in raw_cache:
            raw_cache[raw_key] = _raw_lines(raw_root, case)
        rows.append(
            CaseScore(
                case=case,
                gold=gold,
                prediction=prediction,
                evidence=score_evidence(case, gold, prediction, raw_cache[raw_key]),
                culprit_hit=culprit_match(gold.culprit_file, prediction.culprit_file),
            )
        )

    by_repo: dict[str, list[CaseScore]] = defaultdict(list)
    for row in rows:
        by_repo[row.case.repo].append(row)
    per_repo = {
        repo: _aggregate_rows(repo_rows, seed=seed, resamples=resamples)
        for repo, repo_rows in sorted(by_repo.items())
    }
    return BenchResult(
        system=system_name,
        split=split,
        labeler=labeler,
        git_commit=_git_commit(),
        seed=seed,
        resamples=resamples,
        date=datetime.now(UTC).date().isoformat(),
        n_labeled=len(rows),
        candidate_cases=selection.candidate_cases,
        skipped_no_label=selection.candidate_cases - len(selection.pairs),
        metadata_mismatches=selection.metadata_mismatches,
        rows=tuple(rows),
        overall=_aggregate_rows(rows, seed=seed, resamples=resamples),
        per_repo=per_repo,
    )


__all__ = [
    "BenchError",
    "BenchResult",
    "CaseScore",
    "GoldSelection",
    "Split",
    "split_guard",
    "select_gold",
    "run_bench",
]
