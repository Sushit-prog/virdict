from __future__ import annotations

from collections.abc import Iterator

import pytest

from citriage.github.client import GitHubClient


@pytest.fixture
def client() -> Iterator[GitHubClient]:
    with GitHubClient(
        "test-token",
        sleep=lambda _: None,
        max_sleep_seconds=0.0,
    ) as github:
        yield github
