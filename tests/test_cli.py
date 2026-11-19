from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from citriage.cli import _load_env, _token, app


def _write_env(path: Path, token: str) -> None:
    (path / ".env").write_text(
        f"GITHUB_TOKEN={token}\n",
        encoding="utf-8",
    )


def test_missing_token_fails_clearly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    result = CliRunner().invoke(app, ["collect", "--repo", "o/r"])

    assert result.exit_code == 1
    assert "GITHUB_TOKEN is not set" in result.output
    assert "Set it in the environment or in .env" in result.output


def test_invalid_repo_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    result = CliRunner().invoke(app, ["collect", "--repo", "bad"])

    assert result.exit_code == 2
    assert "expected owner/name" in result.output
    assert "test-token" not in result.output


def test_dotenv_is_loaded_when_env_var_is_absent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    _write_env(tmp_path, "dotenv-token")

    _load_env()

    assert _token() == "dotenv-token"


def test_environment_variable_overrides_dotenv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_TOKEN", "environment-token")
    _write_env(tmp_path, "dotenv-token")

    _load_env()

    assert _token() == "environment-token"


def test_token_never_appears_in_error_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_TOKEN", "super-secret-token")
    result = CliRunner().invoke(app, ["collect", "--repo", "bad"])

    assert result.exit_code == 2
    assert "super-secret-token" not in result.output
