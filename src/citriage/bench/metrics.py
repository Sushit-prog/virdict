from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from citriage.bench.systems import Prediction
from citriage.cases import Case
from citriage.labels import Label, LabelName
from citriage.parse.logparse import _normalize

BENCH_SEED = 20261120
BOOTSTRAP_RESAMPLES = 2000
CI_LOW_PERCENTILE = 0.025
CI_HIGH_PERCENTILE = 0.975


@dataclass(frozen=True)
class ConfidenceInterval:
    low: float
    high: float

    def __str__(self) -> str:
        return f"[{self.low:.3f}, {self.high:.3f}]"


@dataclass(frozen=True)
class LabelMetrics:
    label: LabelName
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class ConfusionMatrix:
    labels: tuple[LabelName, ...]
    rows: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class ClassificationMetrics:
    n: int
    accuracy: float
    macro_f1: float
    per_label: dict[LabelName, LabelMetrics]
    confusion: ConfusionMatrix


@dataclass(frozen=True)
class EvidenceCaseMetrics:
    hallucinated_citations: int
    total_citations: int
    citation_gold_hits: int
    citation_gold_lines: int
    evidence_recall_ceiling: bool | None


@dataclass(frozen=True)
class AggregateMetrics:
    n: int
    classification: ClassificationMetrics
    accuracy_ci: ConfidenceInterval
    macro_f1_ci: ConfidenceInterval
    culprit_hits: int
    culprit_total: int
    culprit_hit_rate: float | None
    hallucinated_citations: int
    total_citations: int
    hallucinated_citation_rate: float | None
    citation_gold_hits: int
    citation_gold_lines: int
    citation_gold_agreement: float | None
    ceiling_hits: int
    ceiling_total: int
    evidence_recall_ceiling: float | None
    unknown_count: int
    unknown_rate: float
    confident_wrong_count: int
    confident_wrong_rate: float
    tokens_in: int
    tokens_out: int
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    live_call_count: int
    cache_hit_count: int
    cache_hit_rate: float | None
    invalid_output_count: int
    invalid_output_rate: float
    retry_count: int
    retry_rate: float
    prompt_truncated_count: int
    prompt_truncated_rate: float | None


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def _label_key(label: LabelName) -> str:
    return label.value


def classification_metrics(
    gold: Sequence[LabelName],
    predicted: Sequence[LabelName],
) -> ClassificationMetrics:
    if len(gold) != len(predicted):
        raise ValueError("gold and predicted must have the same length")
    gold_support = Counter(gold)
    predicted_support = Counter(predicted)
    active = sorted(
        (
            label
            for label in set(gold) | set(predicted)
            if gold_support[label] > 0 or predicted_support[label] > 0
        ),
        key=_label_key,
    )
    counts = Counter(zip(gold, predicted, strict=True))
    per_label: dict[LabelName, LabelMetrics] = {}
    for label in active:
        tp = counts[(label, label)]
        fp = sum(counts[(other, label)] for other in active if other != label)
        fn = sum(counts[(label, other)] for other in active if other != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = LabelMetrics(
            label=label,
            precision=precision,
            recall=recall,
            f1=f1,
            support=gold_support[label],
        )
    accuracy = _mean([float(g == p) for g, p in zip(gold, predicted, strict=True)])
    macro_f1 = _mean([item.f1 for item in per_label.values()])
    matrix = ConfusionMatrix(
        labels=tuple(active),
        rows=tuple(tuple(counts[(row, column)] for column in active) for row in active),
    )
    return ClassificationMetrics(
        n=len(gold),
        accuracy=accuracy,
        macro_f1=macro_f1,
        per_label=per_label,
        confusion=matrix,
    )


def bootstrap_ci[T](
    values: Sequence[T],
    statistic: Callable[[list[T]], float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BENCH_SEED,
) -> ConfidenceInterval:
    if not values or resamples < 1:
        return ConfidenceInterval(0.0, 0.0)
    rng = random.Random(seed)
    size = len(values)
    scores = [
        statistic([values[rng.randrange(size)] for _ in range(size)]) for _ in range(resamples)
    ]
    scores.sort()
    low_index = int(CI_LOW_PERCENTILE * resamples)
    high_index = max(low_index, int(CI_HIGH_PERCENTILE * resamples) - 1)
    return ConfidenceInterval(scores[low_index], scores[high_index])


def bootstrap_accuracy_ci(
    gold: Sequence[LabelName],
    predicted: Sequence[LabelName],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BENCH_SEED,
) -> ConfidenceInterval:
    values = [float(g == p) for g, p in zip(gold, predicted, strict=True)]
    return bootstrap_ci(values, _mean, resamples=resamples, seed=seed)


def bootstrap_macro_f1_ci(
    gold: Sequence[LabelName],
    predicted: Sequence[LabelName],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BENCH_SEED,
) -> ConfidenceInterval:
    values = list(zip(gold, predicted, strict=True))

    def statistic(sample: list[tuple[LabelName, LabelName]]) -> float:
        left = [pair[0] for pair in sample]
        right = [pair[1] for pair in sample]
        return classification_metrics(left, right).macro_f1

    return bootstrap_ci(values, statistic, resamples=resamples, seed=seed)


def _near_any(line: int, gold_lines: set[int]) -> bool:
    return any(abs(line - gold) <= 3 for gold in gold_lines)


def _raw_line(raw_lines: Sequence[str] | None, line_no: int) -> str | None:
    if raw_lines is None or line_no < 1 or line_no > len(raw_lines):
        return None
    return raw_lines[line_no - 1]


def _quote_matches(text: str, raw_line: str) -> bool:
    quoted = text.strip()
    actual = _normalize(raw_line).strip()
    return bool(quoted and actual and (quoted in actual or actual in quoted))


def score_evidence(
    case: Case,
    gold: Label,
    prediction: Prediction,
    raw_lines: Sequence[str] | None = None,
) -> EvidenceCaseMetrics:
    gold_lines = set(gold.evidence_lines)
    hallucinated = 0
    citation_hits = 0
    citation_lines = 0

    if prediction.cited_lines:
        shown_lines = set(prediction.shown_line_numbers)
        citations: Sequence[int] = prediction.cited_lines
        for line in citations:
            if line not in shown_lines:
                hallucinated += 1
            if gold_lines:
                citation_lines += 1
                if _near_any(line, gold_lines):
                    citation_hits += 1
    else:
        parsed = {item.id: item for item in case.evidence}
        for citation_id in prediction.cited_evidence_ids:
            item = parsed.get(citation_id)
            if item is None:
                hallucinated += 1
                continue
            raw_line = _raw_line(raw_lines, item.line_no)
            if raw_line is not None and not _quote_matches(item.text, raw_line):
                hallucinated += 1
            if gold_lines:
                citation_lines += 1
                if _near_any(item.line_no, gold_lines):
                    citation_hits += 1

    ceiling: bool | None = None
    if gold_lines:
        ceiling = any(
            item.kind != "tail" and _near_any(item.line_no, gold_lines) for item in case.evidence
        )
    total_citations = (
        len(prediction.cited_lines)
        if prediction.cited_lines
        else len(prediction.cited_evidence_ids)
    )
    return EvidenceCaseMetrics(
        hallucinated_citations=hallucinated,
        total_citations=total_citations,
        citation_gold_hits=citation_hits,
        citation_gold_lines=citation_lines,
        evidence_recall_ceiling=ceiling,
    )


def _normalize_path(value: str) -> str:
    path = value.strip().replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    return path


def culprit_match(gold: str | None, predicted: str | None) -> bool | None:
    if gold is None:
        return None
    return _normalize_path(gold) == _normalize_path(predicted or "")


def aggregate_metrics(
    gold: Sequence[LabelName],
    predicted: Sequence[LabelName],
    evidence: Sequence[EvidenceCaseMetrics],
    culprits: Sequence[bool | None],
    *,
    predictions: Sequence[Prediction] | None = None,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BENCH_SEED,
) -> AggregateMetrics:
    classification = classification_metrics(gold, predicted)
    culprit_hits = sum(bool(item) for item in culprits if item is not None)
    culprit_total = sum(item is not None for item in culprits)
    hallucinated = sum(item.hallucinated_citations for item in evidence)
    total_citations = sum(item.total_citations for item in evidence)
    citation_hits = sum(item.citation_gold_hits for item in evidence)
    citation_lines = sum(item.citation_gold_lines for item in evidence)
    ceiling_results = [item.evidence_recall_ceiling for item in evidence]
    ceiling_total = sum(item is not None for item in ceiling_results)
    ceiling_hits = sum(bool(item) for item in ceiling_results if item is not None)
    unknown_count = sum(item == LabelName.UNKNOWN for item in predicted)
    confident_wrong_count = sum(
        g != p and p != LabelName.UNKNOWN for g, p in zip(gold, predicted, strict=True)
    )
    if predictions is None:
        tokens_in = 0
        tokens_out = 0
        latency_p50_ms = None
        latency_p95_ms = None
        live_call_count = 0
        cache_hit_count = 0
        cache_hit_rate = None
        invalid_output_count = 0
        invalid_output_rate = 0.0
        retry_count = 0
        retry_rate = 0.0
        prompt_truncated_count = 0
        prompt_truncated_rate = None
    else:
        tokens_in = sum(item.raw_tokens_in for item in predictions)
        tokens_out = sum(item.raw_tokens_out for item in predictions)
        live_latencies = [item.latency_ms for item in predictions if not item.cached]
        latency_p50_ms = _percentile(live_latencies, 0.5)
        latency_p95_ms = _percentile(live_latencies, 0.95)
        cache_hit_count = sum(item.cached for item in predictions)
        live_call_count = len(predictions) - cache_hit_count
        cache_hit_rate = cache_hit_count / len(predictions) if predictions else None
        invalid_output_count = sum(item.invalid_output for item in predictions)
        invalid_output_rate = invalid_output_count / len(predictions) if predictions else 0.0
        retry_count = sum(item.retry_count for item in predictions)
        retry_rate = retry_count / len(predictions) if predictions else 0.0
        prompt_truncated_count = sum(item.prompt_truncated for item in predictions)
        prompt_truncated_rate = prompt_truncated_count / len(predictions) if predictions else None
    return AggregateMetrics(
        n=len(gold),
        classification=classification,
        accuracy_ci=bootstrap_accuracy_ci(gold, predicted, resamples=resamples, seed=seed),
        macro_f1_ci=bootstrap_macro_f1_ci(gold, predicted, resamples=resamples, seed=seed),
        culprit_hits=culprit_hits,
        culprit_total=culprit_total,
        culprit_hit_rate=culprit_hits / culprit_total if culprit_total else None,
        hallucinated_citations=hallucinated,
        total_citations=total_citations,
        hallucinated_citation_rate=hallucinated / total_citations if total_citations else None,
        citation_gold_hits=citation_hits,
        citation_gold_lines=citation_lines,
        citation_gold_agreement=citation_hits / citation_lines if citation_lines else None,
        ceiling_hits=ceiling_hits,
        ceiling_total=ceiling_total,
        evidence_recall_ceiling=ceiling_hits / ceiling_total if ceiling_total else None,
        unknown_count=unknown_count,
        unknown_rate=unknown_count / len(gold) if gold else 0.0,
        confident_wrong_count=confident_wrong_count,
        confident_wrong_rate=confident_wrong_count / len(gold) if gold else 0.0,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        latency_p50_ms=latency_p50_ms,
        latency_p95_ms=latency_p95_ms,
        live_call_count=live_call_count,
        cache_hit_count=cache_hit_count,
        cache_hit_rate=cache_hit_rate,
        invalid_output_count=invalid_output_count,
        invalid_output_rate=invalid_output_rate,
        retry_count=retry_count,
        retry_rate=retry_rate,
        prompt_truncated_count=prompt_truncated_count,
        prompt_truncated_rate=prompt_truncated_rate,
    )


__all__ = [
    "BENCH_SEED",
    "BOOTSTRAP_RESAMPLES",
    "AggregateMetrics",
    "ClassificationMetrics",
    "ConfidenceInterval",
    "EvidenceCaseMetrics",
    "LabelMetrics",
    "aggregate_metrics",
    "bootstrap_accuracy_ci",
    "bootstrap_ci",
    "bootstrap_macro_f1_ci",
    "classification_metrics",
    "culprit_match",
    "score_evidence",
]
