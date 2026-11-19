from __future__ import annotations

import csv
from datetime import timedelta
from pathlib import Path

import pytest

from citriage.bench.report import write_bench_reports
from citriage.bench.runner import (
    BenchError,
    SystemRunMetadata,
    run_bench,
    select_gold,
    split_guard,
)
from citriage.bench.systems import B0System
from citriage.labels import LabelName
from tests.helpers import (
    DEFAULT_NOW,
    bench_case,
    bench_evidence,
    bench_label,
)


def test_split_guard_requires_confirmation_and_frozen_file(tmp_path: Path) -> None:
    frozen = tmp_path / "PARSER_FROZEN.md"
    frozen.write_text("frozen\n", encoding="utf-8")

    with pytest.raises(BenchError, match="--confirm-test"):
        split_guard("test", False, frozen)

    warning = split_guard("test", True, frozen)
    assert warning is not None
    assert "test labels must not drive parser or B0 changes" in warning

    frozen.unlink()
    with pytest.raises(BenchError, match="PARSER_FROZEN.md"):
        split_guard("test", True, frozen)


def test_select_gold_uses_exact_labeler_and_benchmark_cases() -> None:
    cases = [
        bench_case(case_id="c1", split="dev"),
        bench_case(case_id="c2", split="test"),
        bench_case(case_id="c3", split="dev", in_benchmark=False),
    ]
    labels = [
        bench_label(case_id="c1", labeler="alice", id="l1"),
        bench_label(
            case_id="c1",
            labeler="alice-r2",
            id="l2",
            label=LabelName.DEPENDENCY,
        ),
    ]

    selection = select_gold(cases, labels, "dev", "alice")

    assert [case.case_id for case, _ in selection.pairs] == ["c1"]
    assert selection.candidate_cases == 1
    assert selection.metadata_mismatches == 0


def test_select_gold_treats_undo_as_no_current_label() -> None:
    later = DEFAULT_NOW + timedelta(seconds=1)
    labels = [
        bench_label(case_id="c1", labeler="alice", id="l1"),
        bench_label(
            case_id="c1",
            labeler="alice",
            action="undo",
            label=None,
            id="l2",
            labeled_at=later,
        ),
    ]

    selection = select_gold([bench_case()], labels, "dev", "alice")

    assert selection.pairs == ()


def test_select_gold_rejects_r2_labeler() -> None:
    with pytest.raises(BenchError, match="-r2"):
        select_gold([], [], "dev", "alice-r2")


def test_run_bench_fails_with_empty_gold(tmp_path: Path) -> None:
    case = bench_case()
    cases_path = tmp_path / "cases.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    cases_path.write_text(case.model_dump_json() + "\n", encoding="utf-8")
    labels_path.write_text("", encoding="utf-8")

    with pytest.raises(BenchError, match="no current gold labels"):
        run_bench(
            system=B0System(),
            system_name="b0",
            split="dev",
            labeler="alice",
            cases_path=cases_path,
            labels_path=labels_path,
            raw_root=tmp_path / "raw",
            seed=7,
            resamples=10,
        )


def test_run_bench_scores_and_writes_reports(tmp_path: Path) -> None:
    evidence = bench_evidence(kind="pytest_summary")
    case = bench_case(evidence=[evidence])
    gold = bench_label(
        evidence_lines=[10],
        culprit_file="tests/test_x.py",
        label=LabelName.TEST_FAILURE,
    )
    cases_path = tmp_path / "cases.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    cases_path.write_text(case.model_dump_json() + "\n", encoding="utf-8")
    labels_path.write_text(gold.model_dump_json() + "\n", encoding="utf-8")
    log_path = tmp_path / "raw" / "o__r" / "1" / "job_100.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text(evidence.text + "\n", encoding="utf-8")

    result = run_bench(
        system=B0System(),
        system_name="b0",
        split="dev",
        labeler="alice",
        cases_path=cases_path,
        labels_path=labels_path,
        raw_root=tmp_path / "raw",
        seed=7,
        resamples=10,
        metadata=SystemRunMetadata(
            model="fake-model",
            model_digest="fake-digest",
            num_ctx=2048,
            prompt_version="b1-v1",
        ),
    )
    csv_path, markdown_path = write_bench_reports(result, tmp_path / "reports")

    assert result.n_labeled == 1
    assert result.rows[0].prediction.label == LabelName.TEST_FAILURE
    assert result.overall.classification.accuracy == 1.0
    assert result.rows[0].culprit_hit is True
    assert csv_path.name == "b0_dev.csv"
    assert markdown_path.name == "b0_dev.md"
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "git commit:" in markdown
    assert "seed: 7" in markdown
    assert "N labeled: 1" in markdown
    assert "hallucinated_citation_rate" in markdown
    assert "citation_gold_agreement" in markdown
    assert "model: fake-model" in markdown
    assert "model digest: fake-digest" in markdown
    assert "num_ctx: 2048" in markdown
    assert "prompt version: b1-v1" in markdown
    assert "latency p50 ms (live)" in markdown
    assert "cache-hit rate" in markdown

    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["model"] == "fake-model"
    assert rows[0]["prompt_version"] == "b1-v1"
    assert "tokens_in" in rows[0]
    assert "latency_p95_ms" in rows[0]
    assert "invalid_output_rate" in rows[0]
    assert "retry_rate" in rows[0]
