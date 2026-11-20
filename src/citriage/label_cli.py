from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal, TextIO

import typer
from pydantic import BaseModel, ConfigDict, Field

from citriage.cases import CASES_PATH, Case, load_cases
from citriage.labels import (
    LABELS_PATH,
    Label,
    LabelName,
    append_label,
    base_labeler,
    covered_labelers,
    latest_labels,
    load_labels,
)
from citriage.storage import log_path

QUEUE_PATH = Path("data/labels/queue.json")
QUEUE_SEED = "citriage-label-v1"
RAW_ROOT = Path("data/raw")
SPLIT_MAINS = {"dev": 30, "test": 60}
FLAKE_MINIMUM = 15
TOOL_MINIMUM = 10
TOOL_NAMES = {"ruff", "mypy", "pyright", "basedpyright", "pip", "uv", "poetry"}
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\s*")
WORKER_RE = re.compile(r"^\[[^\]]+\]\s*")
PROGRESS_RE = re.compile(r"^\[\s*\d+%\]\s*")
EXIT_CODE_RE = re.compile(r"^##\[error\]Process completed with exit code \d+\.?$")
SHORT_SUMMARY_RE = re.compile(r"short test summary info")
PYTEST_FAILURE_RE = re.compile(r"^(FAILED|ERROR)\b")
PYTEST_TOTAL_RE = re.compile(r"^=+.*\b\d+\s+(?:failed|errors?|passed)\b.*=+$")
DISPLAY_LINE_LIMIT = 300
LABEL_CHOICES = list(LabelName)
SelectionReason = Literal["flake_minimum", "tool_minimum", "quota"]
label_app = typer.Typer(add_completion=False, no_args_is_help=True)


class QueueEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    repo: str
    run_id: int
    split: Literal["dev", "test"]
    workflow_name: str
    step: str
    signature_rank: int | None
    flake_confirmed: bool
    tools_detected: list[str]
    selection_reason: SelectionReason


class LabelPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: str
    entries: list[QueueEntry]
    relaxations: list[str] = Field(default_factory=list)
    shortfalls: list[str] = Field(default_factory=list)


def _hash_key(seed: str, case_id: str) -> str:
    return hashlib.sha1(f"{seed}:{case_id}".encode()).hexdigest()


def _case_sort_key(seed: str, case: Case) -> tuple[str, str, str]:
    return (_hash_key(seed, case.case_id), case.repo, case.step)


def _has_target_tool(case: Case) -> bool:
    return bool(TOOL_NAMES.intersection(case.tools_detected))


def _select_attempt(
    cases: list[Case],
    seed: str,
    rank_limit: int,
    repo_cap: int,
    split_targets: dict[str, int],
    flake_minimum: int,
    tool_minimum: int,
) -> tuple[list[QueueEntry], list[str]]:
    eligible = [
        case
        for case in cases
        if case.signature_rank is not None and case.signature_rank <= rank_limit
    ]
    selected: dict[str, QueueEntry] = {}
    split_counts: Counter[str] = Counter()
    repo_counts: Counter[tuple[str, str]] = Counter()

    def take(case: Case, reason: SelectionReason) -> bool:
        split = case.split
        if split_counts[split] >= split_targets[split]:
            return False
        key = (case.repo, split)
        if repo_counts[key] >= repo_cap:
            return False
        selected[case.case_id] = QueueEntry(
            case_id=case.case_id,
            repo=case.repo,
            run_id=case.run_id,
            split=split,
            workflow_name=case.workflow_name,
            step=case.step,
            signature_rank=case.signature_rank,
            flake_confirmed=case.flake_confirmed,
            tools_detected=case.tools_detected,
            selection_reason=reason,
        )
        split_counts[split] += 1
        repo_counts[key] += 1
        return True

    for case in sorted(
        (case for case in eligible if case.flake_confirmed),
        key=lambda case: _case_sort_key(seed, case),
    ):
        if sum(entry.flake_confirmed for entry in selected.values()) >= flake_minimum:
            break
        take(case, "flake_minimum")

    for case in sorted(
        (case for case in eligible if _has_target_tool(case)),
        key=lambda case: _case_sort_key(seed, case),
    ):
        if case.case_id in selected:
            continue
        if (
            sum(bool(TOOL_NAMES.intersection(entry.tools_detected)) for entry in selected.values())
            >= tool_minimum
        ):
            break
        take(case, "tool_minimum")

    for case in sorted(
        eligible,
        key=lambda case: (case.signature_rank or 0, _case_sort_key(seed, case)),
    ):
        if case.case_id in selected:
            continue
        take(case, "quota")

    entries = [selected[case.case_id] for case in eligible if case.case_id in selected]
    shortfalls: list[str] = []
    for split, target in split_targets.items():
        shortfall = max(0, target - split_counts[split])
        if shortfall:
            shortfalls.append(f"{split} short by {shortfall}")
    flake_count = sum(entry.flake_confirmed for entry in entries)
    if flake_count < flake_minimum:
        shortfalls.append(f"flake_confirmed short by {flake_minimum - flake_count}")
    tool_count = sum(bool(TOOL_NAMES.intersection(entry.tools_detected)) for entry in entries)
    if tool_count < tool_minimum:
        shortfalls.append(f"tool coverage short by {tool_minimum - tool_count}")
    return entries, shortfalls


def plan_queue(
    cases: list[Case],
    path: Path = QUEUE_PATH,
    *,
    seed: str = QUEUE_SEED,
    split_targets: dict[str, int] | None = None,
    flake_minimum: int = FLAKE_MINIMUM,
    tool_minimum: int = TOOL_MINIMUM,
) -> LabelPlan:
    targets = dict(SPLIT_MAINS if split_targets is None else split_targets)
    bench = [case for case in cases if case.in_benchmark]
    attempts = [(2, 12), (3, 12), (3, 15)]
    relaxations: list[str] = []
    entries: list[QueueEntry] = []
    shortfalls: list[str] = []
    previous_rank = 2
    previous_cap = 12

    for rank_limit, repo_cap in attempts:
        if rank_limit != previous_rank:
            relaxations.append(f"signature_rank from <={previous_rank} to <={rank_limit}")
        if repo_cap != previous_cap:
            relaxations.append(f"per-repo cap from {previous_cap} to {repo_cap}")
        previous_rank, previous_cap = rank_limit, repo_cap
        entries, shortfalls = _select_attempt(
            bench,
            seed,
            rank_limit,
            repo_cap,
            targets,
            flake_minimum,
            tool_minimum,
        )
        split_short = any(
            shortfall.startswith(f"{split} short") for split in targets for shortfall in shortfalls
        )
        if not split_short:
            break

    plan = LabelPlan(
        seed=seed,
        entries=entries,
        relaxations=relaxations,
        shortfalls=shortfalls,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
    return plan


def load_queue(path: Path = QUEUE_PATH) -> LabelPlan:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return LabelPlan.model_validate(raw)


def _print_plan_summary(plan: LabelPlan) -> None:
    splits = Counter(entry.split for entry in plan.entries)
    repos = Counter((entry.repo, entry.split) for entry in plan.entries)
    flakes = sum(entry.flake_confirmed for entry in plan.entries)
    tools = sum(bool(TOOL_NAMES.intersection(entry.tools_detected)) for entry in plan.entries)
    reasons = Counter(entry.selection_reason for entry in plan.entries)
    print(f"seed: {plan.seed}")
    print(f"queued: {len(plan.entries)} (dev={splits['dev']}, test={splits['test']})")
    print(f"flake_confirmed: {flakes}, target-tool cases: {tools}")
    print("selection reasons:", dict(sorted(reasons.items())))
    repo_summary = {f"{repo}/{split}": count for (repo, split), count in sorted(repos.items())}
    print("per repo/split:", repo_summary)
    print("relaxations:", plan.relaxations or ["none"])
    print("shortfalls:", plan.shortfalls or ["none"])


@label_app.command("plan")
def label_plan(
    cases_path: Annotated[Path, typer.Option("--cases")] = CASES_PATH,
    out: Annotated[Path, typer.Option("--out")] = QUEUE_PATH,
    seed: Annotated[str, typer.Option("--seed")] = QUEUE_SEED,
) -> None:
    _configure_console()
    plan = plan_queue(load_cases(cases_path), out, seed=seed)
    _print_plan_summary(plan)
    print(f"wrote {out}")


def _configure_console() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")


def _display_text(line: str) -> str:
    return ANSI_RE.sub("", line)


def _normalized_log_line(line: str) -> str:
    text = TIMESTAMP_RE.sub("", _display_text(line))
    text = WORKER_RE.sub("", text)
    text = PROGRESS_RE.sub("", text)
    return text.strip()


def _format_log_line(number: int, line: str) -> str:
    text = _display_text(line)
    if len(text) > DISPLAY_LINE_LIMIT:
        text = text[: DISPLAY_LINE_LIMIT - 3] + "..."
    return f"{number:6d} | {text}"


def _raw_log_lines(case: Case, raw_root: Path = RAW_ROOT) -> tuple[Path, list[str], int]:
    path = log_path(
        raw_root,
        case.repo,
        case.run_id,
        case.job_id,
        run_dir_name=case.run_dir_name or None,
    )
    if not path.exists():
        return path, [], 0
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    norms = [_normalized_log_line(line) for line in lines]
    anchors = [index for index, line in enumerate(norms) if EXIT_CODE_RE.match(line)]
    anchor = anchors[-1] if anchors else len(lines)
    return path, lines, anchor


def _context_chunk_numbers(
    lines: list[str], anchor: int, chunk_size: int, chunk_index: int
) -> list[int]:
    if chunk_size <= 0 or chunk_index < 0:
        return []
    chunk_end = max(0, anchor - chunk_size * chunk_index)
    chunk_start = max(0, chunk_end - chunk_size)
    if chunk_start >= chunk_end or chunk_start >= len(lines):
        return []
    return list(range(chunk_start + 1, min(chunk_end, len(lines)) + 1))


def _pinned_log_evidence(lines: list[str]) -> list[str]:
    norms = [_normalized_log_line(line) for line in lines]
    ordered: list[int] = []
    selected: set[int] = set()

    def add(index: int) -> None:
        if index not in selected:
            selected.add(index)
            ordered.append(index)

    summary_indices = [index for index, text in enumerate(norms) if SHORT_SUMMARY_RE.search(text)]
    if summary_indices:
        start = summary_indices[-1]
        add(start)
        for index in range(start + 1, len(lines)):
            text = norms[index]
            if not text or PYTEST_TOTAL_RE.match(text):
                break
            if PYTEST_FAILURE_RE.match(text):
                add(index)
    for index, text in enumerate(norms):
        if PYTEST_FAILURE_RE.match(text):
            add(index)
    total_indices = [index for index, text in enumerate(norms) if PYTEST_TOTAL_RE.match(text)]
    if total_indices:
        add(total_indices[-1])
    return [_format_log_line(index + 1, lines[index]) for index in ordered]


def render_case(
    case: Case,
    raw_root: Path = RAW_ROOT,
    context_lines: int = 40,
    context_chunk: int = 0,
) -> str:
    path, lines, anchor = _raw_log_lines(case, raw_root)
    numbers = _context_chunk_numbers(lines, anchor, context_lines, context_chunk)
    context = [_format_log_line(number, lines[number - 1]) for number in numbers]
    evidence = _pinned_log_evidence(lines)
    return "\n".join(
        [
            f"case_id: {case.case_id}",
            f"repo: {case.repo}",
            f"workflow: {case.workflow_name}",
            f"failed steps: {', '.join(case.failed_step_names)}",
            f"html_url: {case.html_url}",
            f"raw log: {path}",
            "pinned log evidence:",
            *(evidence or ["  (none)"]),
            f"log context chunk {context_chunk}:",
            *(context or ["  (no log text available)"]),
        ]
    )


def _parse_evidence_lines(raw: str) -> list[int]:
    if not raw.strip():
        return []
    parts = [part.strip() for part in raw.split(",")]
    values: list[int] = []
    for part in parts:
        if not part.isdigit() or int(part) <= 0:
            raise ValueError("evidence lines must be positive integers separated by commas")
        values.append(int(part))
    return sorted(set(values))


def _discard_pending_input(stream: TextIO | None = None) -> None:
    stdin = stream if stream is not None else sys.stdin
    if not stdin.isatty():
        return
    platform = str(sys.platform)
    if platform == "win32":
        import msvcrt

        while msvcrt.kbhit():
            msvcrt.getwch()
        return
    import select

    fd = stdin.fileno()
    while select.select([fd], [], [], 0)[0]:
        if stdin.readline() == "":
            break


def _prompt(label: str) -> str:
    raw = input(label)
    _discard_pending_input()
    return raw.splitlines()[0].strip() if raw else ""


def _normalize_command(raw: str) -> str:
    return raw.strip().upper()


def _make_record(
    case: Case,
    labeler: str,
    label: LabelName,
    culprit_file: str | None,
    culprit_test: str | None,
    evidence_lines: list[int],
    notes: str,
) -> Label:
    return Label(
        case_id=case.case_id,
        repo=case.repo,
        run_id=case.run_id,
        job_id=case.job_id,
        label=label,
        flaky=True if case.flake_confirmed else None,
        culprit_file=culprit_file,
        culprit_test=culprit_test,
        evidence_lines=evidence_lines,
        notes=notes,
        labeler=labeler,
        labeled_at=datetime.now(UTC),
    )


def _make_undo_record(record: Label, labeler: str) -> Label:
    return Label(
        case_id=record.case_id,
        repo=record.repo,
        run_id=record.run_id,
        job_id=record.job_id,
        action="undo",
        label=None,
        supersedes=record.id,
        labeler=labeler,
        labeled_at=datetime.now(UTC),
    )


def _make_pending_record(record: Label, labeler: str) -> Label:
    return Label(
        case_id=record.case_id,
        repo=record.repo,
        run_id=record.run_id,
        job_id=record.job_id,
        action="pending",
        label=None,
        supersedes=record.id,
        labeler=labeler,
        labeled_at=datetime.now(UTC),
    )


@label_app.command("next")
def label_next(
    split: Annotated[str, typer.Option("--split", help="Queue split: dev or test.")] = "dev",
    labeler: Annotated[str, typer.Option("--labeler")] = os.environ.get("LABELER", "labeler"),
    confirm_test: Annotated[
        bool,
        typer.Option("--confirm-test", help="Required when labeling the test split."),
    ] = False,
    cases_path: Annotated[Path, typer.Option("--cases")] = CASES_PATH,
    queue_path: Annotated[Path, typer.Option("--queue")] = QUEUE_PATH,
    labels_path: Annotated[Path, typer.Option("--labels")] = LABELS_PATH,
    raw_root: Annotated[Path, typer.Option("--raw")] = RAW_ROOT,
) -> None:
    _configure_console()
    if split not in ("dev", "test"):
        typer.secho("--split must be dev or test", err=True, fg=typer.colors.RED)
        raise typer.Exit(2)
    if split == "test":
        typer.secho(
            "WARNING: test labels must not drive parser or B0 changes.",
            err=True,
            fg=typer.colors.YELLOW,
        )
        if not confirm_test:
            typer.secho("Use --confirm-test to label the test split.", err=True)
            raise typer.Exit(2)

    cases = {case.case_id: case for case in load_cases(cases_path)}
    queue_ids = [entry.case_id for entry in load_queue(queue_path).entries if entry.split == split]
    covered = covered_labelers(load_labels(labels_path))
    target = base_labeler(labeler)
    remaining = [case_id for case_id in queue_ids if target not in covered.get(case_id, set())]
    last_record: Label | None = None

    for case_id in remaining:
        case = cases.get(case_id)
        if case is None:
            print(f"skip missing case {case_id}")
            continue
        context_size = 40
        context_chunk = 0
        print()
        print(render_case(case, raw_root, context_lines=context_size, context_chunk=context_chunk))
        while True:
            command = _normalize_command(_prompt("label [1-7], m=more, s=skip, u=undo, q=quit> "))
            if command == "M":
                _, lines, anchor = _raw_log_lines(case, raw_root)
                if _context_chunk_numbers(lines, anchor, context_size, context_chunk + 1):
                    context_chunk += 1
                    print(
                        render_case(
                            case,
                            raw_root,
                            context_lines=context_size,
                            context_chunk=context_chunk,
                        )
                    )
                else:
                    print("start of log reached")
                continue
            if command == "S":
                break
            if command == "Q":
                print("Stopped.")
                return
            if command == "U":
                if last_record is None:
                    print("Nothing to undo.")
                    continue
                undo = _make_undo_record(last_record, labeler)
                append_label(undo, labels_path)
                last_record = None
                print(f"undid {undo.supersedes}")
                break
            if not command.isdigit() or not 1 <= int(command) <= len(LABEL_CHOICES):
                print("Invalid command. Valid: 1-7, M, S, U, Q.")
                continue
            label = LABEL_CHOICES[int(command) - 1]
            culprit_file_raw = _prompt("culprit_file (blank for null): ")
            culprit_test_raw = _prompt("culprit_test (blank for null): ")
            evidence_raw = _prompt(
                "evidence line numbers from displayed log (comma-separated, blank for none): "
            )
            try:
                evidence_lines = _parse_evidence_lines(evidence_raw)
                record = _make_record(
                    case,
                    labeler,
                    label,
                    culprit_file_raw or None,
                    culprit_test_raw or None,
                    evidence_lines,
                    _prompt("notes: "),
                )
            except ValueError as exc:
                print(f"Invalid input: {exc}")
                continue
            append_label(record, labels_path)
            last_record = record
            label_value = record.label.value if record.label is not None else "unknown"
            print(f"saved {record.id} label={label_value}")
            break

    print(f"done; {len(remaining)} queued cases offered for split={split}")


@label_app.command("stats")
def label_stats(
    labeler: Annotated[str | None, typer.Option("--labeler")] = None,
    cases_path: Annotated[Path, typer.Option("--cases")] = CASES_PATH,
    queue_path: Annotated[Path, typer.Option("--queue")] = QUEUE_PATH,
    labels_path: Annotated[Path, typer.Option("--labels")] = LABELS_PATH,
) -> None:
    _configure_console()
    labels = load_labels(labels_path)
    latest = latest_labels(labels)
    target = base_labeler(labeler) if labeler else None
    records = [
        record
        for (case_id, record_labeler), record in latest.items()
        if target is None or base_labeler(record_labeler) == target
    ]
    cases = {case.case_id: case for case in load_cases(cases_path)}
    labels_count = Counter(record.label.value for record in records if record.label is not None)
    labelers_count = Counter(base_labeler(record.labeler) for record in records)
    split_repo: Counter[str] = Counter()
    for record in records:
        case = cases.get(record.case_id)
        if case is not None:
            split_repo[f"{case.split}/{case.repo}"] += 1

    print(f"latest labels: {len(records)}")
    print("by label:", dict(sorted(labels_count.items())))
    print("by labeler:", dict(sorted(labelers_count.items())))
    print("by split/repo:", dict(sorted(split_repo.items())))

    covered = covered_labelers(labels)
    queue_ids = [entry.case_id for entry in load_queue(queue_path).entries]
    if target is None:
        remaining_ids = [case_id for case_id in queue_ids if not covered.get(case_id)]
    else:
        remaining_ids = [
            case_id for case_id in queue_ids if target not in covered.get(case_id, set())
        ]
    print(f"queue remaining: {len(remaining_ids)}/{len(queue_ids)}")
    for case_id in remaining_ids[:10]:
        print(f"  {case_id}")


@label_app.command("relabel")
def label_relabel(
    n: Annotated[int, typer.Option("--n", min=1)] = 20,
    seed: Annotated[str, typer.Option("--seed")] = QUEUE_SEED,
    labels_path: Annotated[Path, typer.Option("--labels")] = LABELS_PATH,
) -> None:
    _configure_console()
    latest = latest_labels(load_labels(labels_path))
    candidates = sorted(
        latest.values(),
        key=lambda record: (_hash_key(seed, record.case_id), record.id),
    )
    for record in candidates[:n]:
        pending = _make_pending_record(record, f"{base_labeler(record.labeler)}-r2")
        append_label(pending, labels_path)
    print(f"queued {min(n, len(candidates))} cases for second-round labeling")


@label_app.command("agreement")
def label_agreement(
    a: Annotated[str, typer.Option("--a", help="First labeler.")],
    b: Annotated[str | None, typer.Option("--b", help="Second labeler.")] = None,
    labels_path: Annotated[Path, typer.Option("--labels")] = LABELS_PATH,
) -> None:
    _configure_console()
    second = b or f"{a}-r2"
    latest = latest_labels(load_labels(labels_path))
    first_by_case = {
        case_id: record.label
        for (case_id, labeler), record in latest.items()
        if labeler == a and record.label is not None
    }
    second_by_case = {
        case_id: record.label
        for (case_id, labeler), record in latest.items()
        if labeler == second and record.label is not None
    }
    shared = sorted(set(first_by_case) & set(second_by_case))
    first_labels = [first_by_case[case_id] for case_id in shared]
    second_labels = [second_by_case[case_id] for case_id in shared]
    if not shared:
        print(f"no shared cases for {a} and {second}")
        return
    kappa = cohens_kappa(first_labels, second_labels)
    raw = sum(
        first == second for first, second in zip(first_labels, second_labels, strict=True)
    ) / len(shared)
    print(f"labelers: {a} vs {second}")
    print(f"shared cases: {len(shared)}")
    print(f"raw agreement: {raw:.3f}")
    print(f"cohen's kappa: {kappa:.3f}")
    first_counts = Counter(label.value for label in first_labels)
    second_counts = Counter(label.value for label in second_labels)
    agree_counts = Counter(
        first.value
        for first, second in zip(first_labels, second_labels, strict=True)
        if first == second
    )
    print("per label (first / second / agreed):")
    for label_name in sorted(set(first_counts) | set(second_counts)):
        print(
            f"  {label_name}: {first_counts[label_name]} / "
            f"{second_counts[label_name]} / {agree_counts[label_name]}"
        )


def cohens_kappa(first: list[LabelName], second: list[LabelName]) -> float:
    if not first or len(first) != len(second):
        return 0.0
    total = len(first)
    observed = sum(a == b for a, b in zip(first, second, strict=True)) / total
    first_counts = Counter(first)
    second_counts = Counter(second)
    expected = sum(
        (first_counts[label] / total) * (second_counts[label] / total)
        for label in set(first_counts) | set(second_counts)
    )
    if 1.0 - expected <= 0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


__all__ = [
    "QUEUE_PATH",
    "QUEUE_SEED",
    "LabelPlan",
    "QueueEntry",
    "cohens_kappa",
    "label_app",
    "load_queue",
    "plan_queue",
    "render_case",
]
