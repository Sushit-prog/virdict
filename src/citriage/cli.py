from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Annotated, cast

import typer
from dotenv import load_dotenv

from citriage.bench.metrics import BENCH_SEED, BOOTSTRAP_RESAMPLES
from citriage.bench.report import DEFAULT_REPORT_DIR, write_bench_reports
from citriage.bench.runner import BenchError, Split, run_bench, split_guard
from citriage.bench.systems import get_system
from citriage.cases import CASES_PATH, build_cases
from citriage.collect import collect_repos
from citriage.errors import AuthenticationError, CitriageError, MissingTokenError
from citriage.github.client import GitHubClient
from citriage.label_cli import label_app
from citriage.labels import LABELS_PATH
from citriage.summary import render_summary

app = typer.Typer(add_completion=False, no_args_is_help=True)
cases_app = typer.Typer(add_completion=False, no_args_is_help=True)
app.add_typer(cases_app, name="cases")
app.add_typer(label_app, name="label")


def _load_env() -> None:
    load_dotenv(Path.cwd() / ".env", override=False)


@app.callback()
def _callback() -> None:
    _load_env()


def _token() -> str:
    return os.environ.get("GITHUB_TOKEN", "").strip()


def _normalize_repos(raw: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in raw:
        parts = value.split("/")
        if len(parts) != 2 or not all(parts):
            raise typer.BadParameter(f"Invalid repo {value!r}; expected owner/name.")
        repo = f"{parts[0]}/{parts[1]}"
        key = repo.lower()
        if key not in seen:
            seen.add(key)
            normalized.append(repo)
    return normalized


def main() -> None:
    app()


@app.command()
def collect(
    repo: Annotated[
        list[str],
        typer.Option("--repo", help="Repository in owner/name form. Repeatable."),
    ],
    per_repo: Annotated[
        int,
        typer.Option("--per-repo", min=1, help="Maximum runs to save per repository."),
    ] = 8,
    out: Annotated[
        Path,
        typer.Option("--out", help="Output directory for raw snapshots."),
    ] = Path("data/raw"),
    exclude_workflow: Annotated[
        list[str] | None,
        typer.Option(
            "--exclude-workflow",
            help="Regex matched against workflow name/path. Repeatable.",
        ),
    ] = None,
    no_default_excludes: Annotated[
        bool,
        typer.Option(
            "--no-default-excludes",
            help="Disable the built-in workflow exclusion patterns.",
        ),
    ] = False,
    flakes: Annotated[
        bool,
        typer.Option(
            "--flakes",
            help="Collect failed attempts from runs that later succeeded.",
        ),
    ] = False,
) -> None:
    token = _token()
    if not token:
        typer.secho(
            "GITHUB_TOKEN is not set. Set it in the environment or in .env, then retry.",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    try:
        repos = _normalize_repos(repo)
    except typer.BadParameter as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(2) from exc

    try:
        with GitHubClient(token) as client:
            summary = collect_repos(
                client,
                repos,
                per_repo=per_repo,
                out=out,
                extra_secrets=(token,),
                exclude_patterns=tuple(exclude_workflow or ()),
                include_default_excludes=not no_default_excludes,
                flakes=flakes,
            )
    except MissingTokenError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(1) from exc
    except AuthenticationError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(1) from exc
    except re.error as exc:
        typer.secho(
            f"Invalid --exclude-workflow regex: {exc}",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(2) from exc
    except CitriageError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(1) from exc

    typer.echo(render_summary(summary))

    if summary.stopped_reason is not None:
        raise typer.Exit(2)
    if summary.has_errors:
        raise typer.Exit(1)


@cases_app.command("build")
def cases_build(
    raw: Annotated[
        Path,
        typer.Option("--raw", help="Directory containing raw run snapshots."),
    ] = Path("data/raw"),
    out: Annotated[
        Path,
        typer.Option("--out", help="Output path for the cases index."),
    ] = CASES_PATH,
) -> None:
    build_cases(raw, out)


@app.command("bench")
def bench(
    system: Annotated[str, typer.Option("--system", help="Registered system name.")],
    split: Annotated[str, typer.Option("--split", help="Split to score: dev or test.")],
    labeler: Annotated[str | None, typer.Option("--labeler")] = None,
    confirm_test: Annotated[
        bool,
        typer.Option("--confirm-test", help="Required when benchmarking the test split."),
    ] = False,
    cases: Annotated[Path, typer.Option("--cases")] = CASES_PATH,
    labels: Annotated[Path, typer.Option("--labels")] = LABELS_PATH,
    raw: Annotated[Path, typer.Option("--raw")] = Path("data/raw"),
    report_dir: Annotated[Path, typer.Option("--report-dir")] = DEFAULT_REPORT_DIR,
    seed: Annotated[int, typer.Option("--seed")] = BENCH_SEED,
    resamples: Annotated[int, typer.Option("--resamples", min=1)] = BOOTSTRAP_RESAMPLES,
) -> None:
    if split not in ("dev", "test"):
        typer.secho("--split must be dev or test", err=True, fg=typer.colors.RED)
        raise typer.Exit(2)
    typed_split = cast(Split, split)
    try:
        warning = split_guard(typed_split, confirm_test)
    except BenchError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(2) from exc
    if warning is not None:
        typer.secho(warning, err=True, fg=typer.colors.YELLOW)

    try:
        predictor = get_system(system)
    except KeyError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(2) from exc

    resolved_labeler = labeler or os.environ.get("LABELER", "").strip() or "labeler"
    try:
        result = run_bench(
            system=predictor,
            system_name=system,
            split=typed_split,
            labeler=resolved_labeler,
            cases_path=cases,
            labels_path=labels,
            raw_root=raw,
            confirm_test=confirm_test,
            seed=seed,
            resamples=resamples,
        )
    except BenchError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(2) from exc
    except ValueError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(1) from exc

    print(f"N labeled: {result.n_labeled}")
    csv_path, markdown_path = write_bench_reports(result, report_dir)
    print(f"wrote {csv_path}")
    print(f"wrote {markdown_path}")
