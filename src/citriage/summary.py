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
    headers = ("Repo", "Runs saved", "Jobs saved", "Median log", "Max log")
    rows: list[tuple[str, str, str, str, str]] = []
    for repo in summary.repos:
        rows.append(
            (
                repo.repo,
                str(repo.runs_saved),
                str(repo.jobs_saved),
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
            format_bytes(_median(all_sizes)),
            format_bytes(max(all_sizes, default=0)),
        )
    )

    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    lines = [
        f"{headers[0]:<{widths[0]}}  "
        f"{headers[1]:>{widths[1]}}  "
        f"{headers[2]:>{widths[2]}}  "
        f"{headers[3]:>{widths[3]}}  "
        f"{headers[4]:>{widths[4]}}"
    ]
    lines.append(
        f"{'-' * widths[0]}  "
        f"{'-' * widths[1]}  "
        f"{'-' * widths[2]}  "
        f"{'-' * widths[3]}  "
        f"{'-' * widths[4]}"
    )
    for row in rows:
        lines.append(
            f"{row[0]:<{widths[0]}}  "
            f"{row[1]:>{widths[1]}}  "
            f"{row[2]:>{widths[2]}}  "
            f"{row[3]:>{widths[3]}}  "
            f"{row[4]:>{widths[4]}}"
        )

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
