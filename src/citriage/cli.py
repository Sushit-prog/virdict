from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer

from citriage.collect import collect_repos
from citriage.errors import AuthenticationError, CitriageError, MissingTokenError
from citriage.github.client import GitHubClient
from citriage.summary import render_summary

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.callback()
def _callback() -> None:
    pass


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
) -> None:
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        typer.secho(
            "GITHUB_TOKEN is not set. Export a read-only token and retry.",
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
            )
    except MissingTokenError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(1) from exc
    except AuthenticationError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(1) from exc
    except CitriageError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(1) from exc

    typer.echo(render_summary(summary))

    if summary.stopped_reason is not None:
        raise typer.Exit(2)
    if summary.has_errors:
        raise typer.Exit(1)
