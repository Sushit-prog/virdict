from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from citriage.github.client import GitHubClient


@pytest.fixture(autouse=True)
def _restore_github_token() -> Iterator[None]:
    had_token = "GITHUB_TOKEN" in os.environ
    original = os.environ.get("GITHUB_TOKEN")
    yield
    if had_token:
        os.environ["GITHUB_TOKEN"] = original or ""
    else:
        os.environ.pop("GITHUB_TOKEN", None)


@pytest.fixture
def client() -> Iterator[GitHubClient]:
    with GitHubClient(
        "test-token",
        sleep=lambda _: None,
        max_sleep_seconds=0.0,
    ) as github:
        yield github
