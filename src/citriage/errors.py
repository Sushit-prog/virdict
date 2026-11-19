from __future__ import annotations

from datetime import datetime


class CitriageError(Exception):
    pass


class MissingTokenError(CitriageError):
    pass


class GitHubAPIError(CitriageError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        url: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.url = url


class AuthenticationError(GitHubAPIError):
    pass


class RateLimitExhausted(CitriageError):
    def __init__(self, reset_at: datetime | None) -> None:
        self.reset_at = reset_at
        if reset_at is None:
            message = "GitHub rate limit exhausted; reset time is unknown."
        else:
            message = f"GitHub rate limit exhausted; resets at {reset_at.isoformat()}."
        super().__init__(message)


class NetworkError(CitriageError):
    pass


class LogUnavailableError(GitHubAPIError):
    pass
