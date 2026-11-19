from __future__ import annotations

from collections import Counter
from pathlib import Path

from citriage.cases import Case, load_cases

RAW = Path("data/raw")
REPORT = Path("reports/parse_coverage.md")


def _log_path(case: Case) -> Path | None:
    for path in RAW.rglob(f"job_{case.job_id}.log"):
        if path.parent.name == str(case.run_id) or path.parent.name.startswith(f"{case.run_id}_a"):
            return path
    return None


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
    cases = load_cases()
    zeros = [case for case in cases if not case.evidence]
    kinds = Counter(item.kind for case in cases for item in case.evidence)
    labels = Counter(case.b0_label.value for case in cases)
    excerpts = _pytest_excerpts(cases)

    lines = [
        "# Parse coverage",
        "",
        f"- Cases: {len(cases)}",
        f"- Cases with zero evidence: {len(zeros)}",
        f"- Distinct evidence kinds: {len(kinds)}",
        "",
        "## Evidence kinds",
        "",
        "| kind | count |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {kind} | {count} |" for kind, count in kinds.most_common())

    lines.extend(["", "## B0 labels", "", "| label | count |", "| --- | ---: |"])
    lines.extend(f"| {label} | {count} |" for label, count in labels.most_common())

    lines.extend(
        [
            "",
            "## Zero-evidence cases (first 15)",
            "",
            "| repo | run_id | job | step |",
            "| --- | ---: | --- | --- |",
        ]
    )
    for case in zeros[:15]:
        lines.append(
            f"| {case.repo} | {case.run_id} | {case.job_name} | {case.step.replace('|', '\\|')} |"
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

    print(f"cases={len(cases)} zero_evidence={len(zeros)}")
    print("evidence kinds:", dict(kinds.most_common()))
    print("B0 labels:", dict(labels.most_common()))
    print("zero-evidence cases (first 15):")
    for case in zeros[:15]:
        print(f"  {case.repo} | {case.run_id} | {case.job_name} | {case.step}")
    print(f"wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
