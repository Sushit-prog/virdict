from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from citriage.cli import app
from citriage.labels import LabelName
from tests.helpers import bench_case, bench_evidence, bench_label


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    evidence = bench_evidence(kind="pytest_summary")
    case = bench_case(evidence=[evidence])
    gold = bench_label(evidence_lines=[10], label=LabelName.TEST_FAILURE)
    cases_path = tmp_path / "cases.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    raw_root = tmp_path / "raw"
    cases_path.write_text(case.model_dump_json() + "\n", encoding="utf-8")
    labels_path.write_text(gold.model_dump_json() + "\n", encoding="utf-8")
    log_path = raw_root / "o__r" / "1" / "job_100.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text(evidence.text + "\n", encoding="utf-8")
    return cases_path, labels_path, raw_root


def test_bench_cli_exact_syntax_writes_dev_reports(tmp_path: Path) -> None:
    cases_path, labels_path, raw_root = _write_inputs(tmp_path)
    report_dir = tmp_path / "reports"

    result = CliRunner().invoke(
        app,
        [
            "bench",
            "--system",
            "b0",
            "--split",
            "dev",
            "--labeler",
            "alice",
            "--cases",
            str(cases_path),
            "--labels",
            str(labels_path),
            "--raw",
            str(raw_root),
            "--report-dir",
            str(report_dir),
            "--seed",
            "7",
            "--resamples",
            "10",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "N labeled: 1" in result.output
    assert (report_dir / "b0_dev.csv").exists()
    assert (report_dir / "b0_dev.md").exists()
    assert not (report_dir / "b0_test.csv").exists()


def test_bench_cli_rejects_test_split_without_confirmation(tmp_path: Path) -> None:
    cases_path, labels_path, raw_root = _write_inputs(tmp_path)
    report_dir = tmp_path / "reports"

    result = CliRunner().invoke(
        app,
        [
            "bench",
            "--system",
            "b0",
            "--split",
            "test",
            "--labeler",
            "alice",
            "--cases",
            str(cases_path),
            "--labels",
            str(labels_path),
            "--raw",
            str(raw_root),
            "--report-dir",
            str(report_dir),
        ],
    )

    assert result.exit_code == 2
    assert "--confirm-test" in result.output
    assert not report_dir.exists()


def test_bench_cli_rejects_unknown_system(tmp_path: Path) -> None:
    cases_path, labels_path, raw_root = _write_inputs(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "bench",
            "--system",
            "nope",
            "--split",
            "dev",
            "--labeler",
            "alice",
            "--cases",
            str(cases_path),
            "--labels",
            str(labels_path),
            "--raw",
            str(raw_root),
            "--report-dir",
            str(tmp_path / "reports"),
        ],
    )

    assert result.exit_code == 2
    assert "unknown system" in result.output
