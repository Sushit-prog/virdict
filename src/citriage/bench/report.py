from __future__ import annotations

import csv
from pathlib import Path

from citriage.bench.runner import BenchResult

DEFAULT_REPORT_DIR = Path("reports/bench")


def default_report_dir() -> Path:
    return DEFAULT_REPORT_DIR


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _format_rate(numerator: int, denominator: int) -> str:
    return _fmt(numerator / denominator if denominator else None)


def render_markdown(result: BenchResult) -> str:
    overall = result.overall
    classification = overall.classification
    raw_tokens_in = sum(row.prediction.raw_tokens_in for row in result.rows)
    raw_tokens_out = sum(row.prediction.raw_tokens_out for row in result.rows)
    latency_ms = sum(row.prediction.latency_ms for row in result.rows)
    lines = [
        f"# Bench report: {result.system} {result.split}",
        "",
        f"- git commit: {result.git_commit}",
        f"- seed: {result.seed}",
        f"- date: {result.date}",
        f"- N labeled: {result.n_labeled}",
        f"- system: {result.system}",
        f"- split: {result.split}",
        f"- labeler: {result.labeler}",
        f"- bootstrap: {result.resamples} resamples, 95% percentile intervals",
        f"- candidate in-benchmark cases: {result.candidate_cases}",
        f"- skipped without current label: {result.skipped_no_label}",
        f"- metadata mismatches skipped: {result.metadata_mismatches}",
        "",
        "## Overall",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| label accuracy | {overall.classification.accuracy:.3f} {overall.accuracy_ci} |",
        f"| macro-F1 | {classification.macro_f1:.3f} {overall.macro_f1_ci} |",
        f"| culprit-file hit rate | {_fmt(overall.culprit_hit_rate)} "
        f"({overall.culprit_hits}/{overall.culprit_total}) |",
        f"| hallucinated_citation_rate | {_fmt(overall.hallucinated_citation_rate)} "
        f"({overall.hallucinated_citations}/{overall.total_citations}) |",
        f"| citation_gold_agreement | {_fmt(overall.citation_gold_agreement)} "
        f"({overall.citation_gold_hits}/{overall.citation_gold_lines}) |",
        f"| evidence recall ceiling | {_fmt(overall.evidence_recall_ceiling)} "
        f"({overall.ceiling_hits}/{overall.ceiling_total}) |",
        f"| unknown rate | {overall.unknown_rate:.3f} ({overall.unknown_count}/{overall.n}) |",
        f"| confident-wrong rate | {overall.confident_wrong_rate:.3f} "
        f"({overall.confident_wrong_count}/{overall.n}) |",
        f"| raw tokens in | {raw_tokens_in} |",
        f"| raw tokens out | {raw_tokens_out} |",
        f"| latency ms | {latency_ms:.3f} |",
        "",
        "## Per-label",
        "",
        "| label | precision | recall | F1 | support |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label in sorted(classification.per_label, key=lambda item: item.value):
        label_metrics = classification.per_label[label]
        lines.append(
            f"| {label.value} | {label_metrics.precision:.3f} | "
            f"{label_metrics.recall:.3f} | {label_metrics.f1:.3f} | "
            f"{label_metrics.support} |"
        )

    labels = classification.confusion.labels
    lines.extend(["", "## Confusion matrix", "", "rows = gold, columns = predicted", ""])
    header = "| gold \\ predicted | " + " | ".join(label.value for label in labels) + " |"
    separator = "| --- | " + " | ".join("---:" for _ in labels) + " |"
    lines.extend([header, separator])
    for label, row in zip(labels, classification.confusion.rows, strict=True):
        values = " | ".join(str(value) for value in row)
        lines.append(f"| {label.value} | {values} |")

    lines.extend(
        [
            "",
            "## Per repo",
            "",
            "| repo | n | accuracy | accuracy CI | macro-F1 | macro-F1 CI | "
            "citation agreement | evidence ceiling |",
            "| --- | ---: | ---: | --- | ---: | --- | ---: | ---: |",
        ]
    )
    for repo, repo_metrics in result.per_repo.items():
        lines.append(
            f"| {repo} | {repo_metrics.n} | {repo_metrics.classification.accuracy:.3f} "
            f"{repo_metrics.accuracy_ci} | {repo_metrics.classification.macro_f1:.3f} "
            f"{repo_metrics.macro_f1_ci} | {_fmt(repo_metrics.citation_gold_agreement)} "
            f"| {_fmt(repo_metrics.evidence_recall_ceiling)} |"
        )
    return "\n".join(lines) + "\n"


def _write_csv(result: BenchResult, path: Path) -> None:
    fieldnames = [
        "git_commit",
        "seed",
        "date",
        "n_labeled",
        "system",
        "split",
        "labeler",
        "case_id",
        "repo",
        "gold",
        "predicted",
        "gold_culprit_file",
        "predicted_culprit_file",
        "culprit_hit",
        "cited_evidence_ids",
        "hallucinated_citations",
        "total_citations",
        "citation_gold_hits",
        "citation_gold_lines",
        "evidence_recall_ceiling",
        "raw_tokens_in",
        "raw_tokens_out",
        "latency_ms",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in result.rows:
            assert row.gold.label is not None
            writer.writerow(
                {
                    "git_commit": result.git_commit,
                    "seed": result.seed,
                    "date": result.date,
                    "n_labeled": result.n_labeled,
                    "system": result.system,
                    "split": result.split,
                    "labeler": result.labeler,
                    "case_id": row.case.case_id,
                    "repo": row.case.repo,
                    "gold": row.gold.label.value,
                    "predicted": row.prediction.label.value,
                    "gold_culprit_file": row.gold.culprit_file or "",
                    "predicted_culprit_file": row.prediction.culprit_file or "",
                    "culprit_hit": "" if row.culprit_hit is None else str(row.culprit_hit),
                    "cited_evidence_ids": "|".join(row.prediction.cited_evidence_ids),
                    "hallucinated_citations": row.evidence.hallucinated_citations,
                    "total_citations": row.evidence.total_citations,
                    "citation_gold_hits": row.evidence.citation_gold_hits,
                    "citation_gold_lines": row.evidence.citation_gold_lines,
                    "evidence_recall_ceiling": ""
                    if row.evidence.evidence_recall_ceiling is None
                    else str(row.evidence.evidence_recall_ceiling),
                    "raw_tokens_in": row.prediction.raw_tokens_in,
                    "raw_tokens_out": row.prediction.raw_tokens_out,
                    "latency_ms": f"{row.prediction.latency_ms:.3f}",
                }
            )


def write_bench_reports(result: BenchResult, report_dir: Path) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / f"{result.system}_{result.split}.csv"
    markdown_path = report_dir / f"{result.system}_{result.split}.md"
    _write_csv(result, csv_path)
    markdown_path.write_text(render_markdown(result), encoding="utf-8", newline="\n")
    return csv_path, markdown_path


__all__ = ["default_report_dir", "render_markdown", "write_bench_reports"]
