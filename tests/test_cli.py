from __future__ import annotations

import pytest
from typer.testing import CliRunner

from citriage.cli import app


def test_missing_token_fails_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    result = CliRunner().invoke(app, ["collect", "--repo", "o/r"])

    assert result.exit_code == 1
    assert "GITHUB_TOKEN is not set" in result.output


def test_invalid_repo_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    result = CliRunner().invoke(app, ["collect", "--repo", "bad"])

    assert result.exit_code == 2
    assert "expected owner/name" in result.output
