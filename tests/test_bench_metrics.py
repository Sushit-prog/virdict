from __future__ import annotations

import pytest

from citriage.bench.metrics import (
    EvidenceCaseMetrics,
    aggregate_metrics,
    bootstrap_accuracy_ci,
    bootstrap_macro_f1_ci,
    classification_metrics,
    culprit_match,
    score_evidence,
)
from citriage.labels import LabelName
from tests.helpers import (
    bench_case,
    bench_evidence,
    bench_label,
    bench_prediction,
)


def test_classification_metrics_tiny_confusion_matrix() -> None:
    gold = [
        LabelName.TEST_FAILURE,
        LabelName.TEST_FAILURE,
        LabelName.DEPENDENCY,
        LabelName.DEPENDENCY,
    ]
    predicted = [
        LabelName.TEST_FAILURE,
        LabelName.DEPENDENCY,
        LabelName.DEPENDENCY,
        LabelName.TEST_FAILURE,
    ]

    metrics = classification_metrics(gold, predicted)

    assert metrics.n == 4
    assert metrics.accuracy == pytest.approx(0.5)
    assert metrics.macro_f1 == pytest.approx(0.5)
    assert metrics.confusion.labels == (LabelName.DEPENDENCY, LabelName.TEST_FAILURE)
    assert metrics.confusion.rows == ((1, 1), (1, 1))
    for item in metrics.per_label.values():
        assert item.f1 == pytest.approx(0.5)
        assert item.support == 2


def test_macro_f1_includes_labels_with_prediction_support_only() -> None:
    gold = [LabelName.TEST_FAILURE, LabelName.LINT_FORMAT]
    predicted = [LabelName.TEST_FAILURE, LabelName.CONFIG_ERROR]

    metrics = classification_metrics(gold, predicted)

    assert set(metrics.per_label) == {
        LabelName.CONFIG_ERROR,
        LabelName.LINT_FORMAT,
        LabelName.TEST_FAILURE,
    }
    assert LabelName.TYPE_ERROR not in metrics.per_label
    assert metrics.per_label[LabelName.CONFIG_ERROR].support == 0
    assert metrics.macro_f1 == pytest.approx(1 / 3)


def test_bootstrap_intervals_are_deterministic_with_fixed_seed() -> None:
    gold = [LabelName.TEST_FAILURE, LabelName.DEPENDENCY, LabelName.UNKNOWN]
    predicted = [LabelName.TEST_FAILURE, LabelName.UNKNOWN, LabelName.UNKNOWN]

    first_accuracy = bootstrap_accuracy_ci(gold, predicted, resamples=50, seed=7)
    second_accuracy = bootstrap_accuracy_ci(gold, predicted, resamples=50, seed=7)
    first_macro = bootstrap_macro_f1_ci(gold, predicted, resamples=50, seed=7)
    second_macro = bootstrap_macro_f1_ci(gold, predicted, resamples=50, seed=7)

    assert first_accuracy == second_accuracy
    assert first_macro == second_macro
    assert first_accuracy.low <= first_accuracy.high
    assert first_macro.low <= first_macro.high


def test_hallucination_and_gold_agreement_are_separate_metrics() -> None:
    case = bench_case(
        evidence=[
            bench_evidence(id="good", line_no=10),
            bench_evidence(id="far", line_no=20, text="different text"),
        ]
    )
    gold = bench_label(evidence_lines=[10])
    prediction = bench_prediction(cited_evidence_ids=["good", "far", "missing"])
    raw_lines = [f"noise {number}" for number in range(1, 21)]
    raw_lines[9] = "FAILED tests/test_x.py::test_a"
    raw_lines[19] = "actual log line"

    metrics = score_evidence(case, gold, prediction, raw_lines)

    assert metrics.hallucinated_citations == 2
    assert metrics.total_citations == 3
    assert metrics.citation_gold_hits == 1
    assert metrics.citation_gold_lines == 2
    assert metrics.evidence_recall_ceiling is True


def test_aggregate_keeps_citation_rates_distinct() -> None:
    evidence = EvidenceCaseMetrics(
        hallucinated_citations=1,
        total_citations=2,
        citation_gold_hits=1,
        citation_gold_lines=1,
        evidence_recall_ceiling=True,
    )

    metrics = aggregate_metrics(
        [LabelName.TEST_FAILURE],
        [LabelName.TEST_FAILURE],
        [evidence],
        [None],
        resamples=10,
    )

    assert metrics.hallucinated_citation_rate == pytest.approx(0.5)
    assert metrics.citation_gold_agreement == pytest.approx(1.0)
    assert metrics.evidence_recall_ceiling == pytest.approx(1.0)


def test_culprit_match_normalizes_paths_and_respects_missing_gold() -> None:
    assert culprit_match("src/x.py", "./src\\x.py") is True
    assert culprit_match("src/x.py", "src/y.py") is False
    assert culprit_match(None, "src/x.py") is None
