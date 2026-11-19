from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from citriage.cases import Case, load_cases
from citriage.storage import log_path

RAW = Path("data/raw")
REPORT = Path("reports/parse_coverage.md")


def _log_path(case: Case) -> Path | None:
    path = log_path(
        RAW,
        case.repo,
        case.run_id,
        case.job_id,
        run_dir_name=case.run_dir_name or None,
    )
    return path if path.exists() else None


def _pytest_excerpts(cases: list[Case]) -> list[tuple[Case, str, list[str]]]:
    excerpts: list[tuple[Case, str, list[str]]] = []
    seen_paths: set[Path] = set()
    for case in cases:
        summary = next((item for item in case.evidence if item.kind == "pytest_summary"), None)
        if summary is None:
            continue
        path = _log_path(case)
        if path is None or path in seen_paths:
            continue
        seen_paths.add(path)
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(0, summary.line_no - 3)
        end = min(len(lines), summary.line_no + 2)
        excerpt = [line.replace("\x1b", "\\x1b")[:220] for line in lines[start:end]]
        excerpts.append((case, str(path), excerpt))
        if len(excerpts) == 3:
            break
    return excerpts


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cases = load_cases()
    bench = [case for case in cases if case.in_benchmark]
    zeros_excluding_tail = [
        case for case in bench if not any(item.kind != "tail" for item in case.evidence)
    ]
    zeros_including_tail = [case for case in bench if not case.evidence]
    kinds_all = Counter(item.kind for case in cases for item in case.evidence)
    kinds_bench = Counter(item.kind for case in bench for item in case.evidence)
    labels_all = Counter(case.b0_label.value for case in cases)
    labels_bench = Counter(case.b0_label.value for case in bench)
    per_repo_split = Counter((case.repo, case.split) for case in cases)
    per_repo_split_bench = Counter((case.repo, case.split) for case in bench)
    excerpts = _pytest_excerpts(bench)

    rate_excluding = len(zeros_excluding_tail) / len(bench) if bench else 0.0
    rate_including = len(zeros_including_tail) / len(bench) if bench else 0.0

    lines = [
        "# Parse coverage",
        "",
        f"- Cases: {len(cases)}",
        f"- In-benchmark cases: {len(bench)}",
        (
            "- In-benchmark zero evidence (excluding tail): "
            f"{len(zeros_excluding_tail)}/{len(bench)} ({rate_excluding:.1%})"
        ),
        (
            "- In-benchmark zero evidence (including tail): "
            f"{len(zeros_including_tail)}/{len(bench)} ({rate_including:.1%})"
        ),
        f"- Distinct evidence kinds: {len(kinds_all)}",
        "",
        "## Cases by repo and split",
        "",
        "| repo | split | all | in_benchmark |",
        "| --- | --- | ---: | ---: |",
    ]
    for (repo, split), count in sorted(per_repo_split.items()):
        lines.append(f"| {repo} | {split} | {count} | {per_repo_split_bench[(repo, split)]} |")

    lines.extend(["", "## Evidence kinds (all cases)", "", "| kind | count |", "| --- | ---: |"])
    lines.extend(f"| {kind} | {count} |" for kind, count in kinds_all.most_common())

    lines.extend(["", "## Evidence kinds (in_benchmark)", "", "| kind | count |", "| --- | ---: |"])
    lines.extend(f"| {kind} | {count} |" for kind, count in kinds_bench.most_common())

    lines.extend(["", "## B0 labels (in_benchmark)", "", "| label | count |", "| --- | ---: |"])
    lines.extend(f"| {label} | {count} |" for label, count in labels_bench.most_common())

    lines.extend(["", "## B0 labels (all)", "", "| label | count |", "| --- | ---: |"])
    lines.extend(f"| {label} | {count} |" for label, count in labels_all.most_common())

    lines.extend(
        [
            "",
            "## Zero-evidence in_benchmark cases (excluding tail, first 15)",
            "",
            "| case_id | repo | run_id | job | step |",
            "| --- | --- | ---: | --- | --- |",
        ]
    )
    for case in zeros_excluding_tail[:15]:
        lines.append(
            f"| {case.case_id} | {case.repo} | {case.run_id} | {case.job_name} | "
            f"{case.step.replace('|', '\\|')} |"
        )

    lines.extend(["", "## Raw pytest excerpts", ""])
    for case, path, excerpt in excerpts:
        lines.extend(
            [
                f"### {case.repo} run {case.run_id} job {case.job_id} ({case.step})",
                "",
                f"Source: `{path}`",
                "",
                "```text",
                *excerpt,
                "```",
                "",
            ]
        )

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    print(f"cases={len(cases)} in_benchmark={len(bench)}")
    print(
        "zero_evidence_excluding_tail="
        f"{len(zeros_excluding_tail)}/{len(bench)} ({rate_excluding:.1%})"
    )
    print(
        "zero_evidence_including_tail="
        f"{len(zeros_including_tail)}/{len(bench)} ({rate_including:.1%})"
    )
    per_repo_split_text = {
        f"{repo}/{split}": count for (repo, split), count in sorted(per_repo_split.items())
    }
    print("per repo/split:", per_repo_split_text)
    print("evidence kinds:", dict(kinds_all.most_common()))
    print("B0 labels in_benchmark:", dict(labels_bench.most_common()))
    print("zero-evidence in_benchmark cases (excluding tail, first 15):")
    for case in zeros_excluding_tail[:15]:
        print(f"  {case.case_id} | {case.repo} | {case.run_id} | {case.job_name} | {case.step}")
    print(f"wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
