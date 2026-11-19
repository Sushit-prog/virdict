from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from citriage.cli import app
from citriage.labels import LabelName
from tests.helpers import bench_case, bench_evidence, bench_label, bench_prediction


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


def test_bench_cli_b1_records_metadata_and_warns_on_truncation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases_path, labels_path, raw_root = _write_inputs(tmp_path)
    report_dir = tmp_path / "reports"

    class _FakeB1:
        model = "fake-model"
        model_digest = "fake-digest"
        num_ctx = 2048
        prompt_version = "b1-v1"

        def predict(self, case: object) -> object:
            return bench_prediction(
                raw_tokens_in=12,
                raw_tokens_out=4,
                latency_ms=25.0,
                prompt_truncated=True,
                cited_lines=(10,),
                shown_line_numbers=(9, 10, 11),
            ).model_copy(
                update={
                    "model": "fake-model",
                    "model_digest": "fake-digest",
                    "num_ctx": 2048,
                    "prompt_version": "b1-v1",
                }
            )

    monkeypatch.setattr("citriage.cli.build_system", lambda name, **kwargs: _FakeB1())
    result = CliRunner().invoke(
        app,
        [
            "bench",
            "--system",
            "b1",
            "--split",
            "dev",
            "--labeler",
            "alice",
            "--model",
            "qwen2.5-coder:1.5b",
            "--num-ctx",
            "2048",
            "--no-cache",
            "--cache-db",
            str(tmp_path / "cache.sqlite3"),
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
    assert "WARNING: prompt truncation" in result.output
    markdown = (report_dir / "b1_dev.md").read_text(encoding="utf-8")
    assert "model: fake-model" in markdown
    assert "model digest: fake-digest" in markdown
    assert "num_ctx: 2048" in markdown
    assert "prompt version: b1-v1" in markdown
    assert "cache-hit rate" in markdown


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
