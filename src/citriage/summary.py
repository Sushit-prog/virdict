from __future__ import annotations

import statistics

from citriage.collect import CollectSummary


def format_bytes(value: int) -> str:
    if value < 1024:
        return f"{value} B"
    if value < 1024 * 1024:
        return f"{value / 1024:.1f} KB"
    return f"{value / (1024 * 1024):.1f} MB"


def _median(values: list[int]) -> int:
    return int(statistics.median(values)) if values else 0


def render_summary(summary: CollectSummary) -> str:
    headers = (
        "Repo",
        "Runs saved",
        "Jobs saved",
        "Candidates evaluated",
        "Excluded",
        "Aggregator-only",
        "Median log",
        "Max log",
    )
    rows: list[tuple[str, ...]] = []
    for repo in summary.repos:
        rows.append(
            (
                repo.repo,
                str(repo.runs_saved),
                str(repo.jobs_saved),
                str(repo.candidates_evaluated),
                str(repo.excluded),
                str(repo.skipped_aggregator_only),
                format_bytes(_median(repo.log_sizes)),
                format_bytes(max(repo.log_sizes, default=0)),
            )
        )

    all_sizes = [size for repo in summary.repos for size in repo.log_sizes]
    rows.append(
        (
            "TOTAL",
            str(sum(repo.runs_saved for repo in summary.repos)),
            str(sum(repo.jobs_saved for repo in summary.repos)),
            str(sum(repo.candidates_evaluated for repo in summary.repos)),
            str(sum(repo.excluded for repo in summary.repos)),
            str(sum(repo.skipped_aggregator_only for repo in summary.repos)),
            format_bytes(_median(all_sizes)),
            format_bytes(max(all_sizes, default=0)),
        )
    )

    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def format_row(row: tuple[str, ...]) -> str:
        return "  ".join(
            f"{cell:<{widths[index]}}" if index == 0 else f"{cell:>{widths[index]}}"
            for index, cell in enumerate(row)
        )

    lines = [format_row(headers), format_row(tuple("-" * width for width in widths))]
    lines.extend(format_row(row) for row in rows)

    remaining = (
        "unknown" if summary.rate_limit_remaining is None else str(summary.rate_limit_remaining)
    )
    lines.append(f"Rate limit remaining: {remaining}")

    if summary.total_skipped:
        lines.append(f"Runs skipped (already saved): {summary.total_skipped}")
    if summary.stopped_reason is not None:
        lines.append(f"Stopped: {summary.stopped_reason}")

    errors = [f"{repo.repo}: {error}" for repo in summary.repos for error in repo.errors]
    if errors:
        lines.append("Errors:")
        lines.extend(f"  - {error}" for error in errors)

    return "\n".join(lines)
