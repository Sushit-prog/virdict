from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

from citriage.cases import CASES_PATH, build_cases
from citriage.collect import collect_repos
from citriage.errors import AuthenticationError, CitriageError, MissingTokenError
from citriage.github.client import GitHubClient
from citriage.summary import render_summary

app = typer.Typer(add_completion=False, no_args_is_help=True)
cases_app = typer.Typer(add_completion=False, no_args_is_help=True)
app.add_typer(cases_app, name="cases")


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
