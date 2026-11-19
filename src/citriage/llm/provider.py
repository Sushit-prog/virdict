from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx
from pydantic import BaseModel, ConfigDict

from citriage.errors import CitriageError

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_CACHE_DB = Path("data/cache/llm.sqlite3")
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_SEED = 20261120
DEFAULT_TEMPERATURE = 0.0
DEFAULT_NUM_CTX = 4096
KEEP_ALIVE = "30s"


class LLMError(CitriageError):
    pass


class LLMUnreachableError(LLMError):
    pass


class LLMTimeoutError(LLMError):
    pass


class LLMBadJSONError(LLMError):
    pass


class LLMMissingConfigError(LLMError):
    pass


class LLMResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    model: str
    model_digest: str
    cached: bool = False


class LLMProvider(Protocol):
    def generate(self, prompt: str, schema: dict[str, Any]) -> LLMResult: ...


@dataclass(frozen=True)
class ProviderSettings:
    model: str
    model_digest: str
    num_ctx: int
    temperature: float
    seed: int
    provider: str
    base_url: str = ""


def _post_json(
    client: httpx.Client,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    try:
        response = client.post(url, json=payload, headers=headers)
        response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise LLMTimeoutError(f"LLM request timed out: {url}") from exc
    except httpx.RequestError as exc:
        raise LLMUnreachableError(f"LLM endpoint unreachable: {url}") from exc
    except httpx.HTTPStatusError as exc:
        raise LLMUnreachableError(
            f"LLM endpoint returned {exc.response.status_code}: {url}"
        ) from exc
    try:
        body = response.json()
    except ValueError as exc:
        raise LLMBadJSONError(f"LLM endpoint returned invalid JSON: {url}") from exc
    if not isinstance(body, dict):
        raise LLMBadJSONError(f"LLM endpoint returned non-object JSON: {url}")
    return body


class OllamaProvider:
    def __init__(
        self,
        model: str,
        *,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        num_ctx: int = DEFAULT_NUM_CTX,
        temperature: float = DEFAULT_TEMPERATURE,
        seed: int = DEFAULT_SEED,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.seed = seed
        self.timeout = timeout
        self._client = client or httpx.Client(timeout=timeout)
        self._digest: str | None = None

    def settings(self) -> ProviderSettings:
        return ProviderSettings(
            model=self.model,
            model_digest=self._ensure_digest(),
            num_ctx=self.num_ctx,
            temperature=self.temperature,
            seed=self.seed,
            provider="ollama",
            base_url=self.base_url,
        )

    def _ensure_digest(self) -> str:
        if self._digest is None:
            body = _post_json(
                self._client,
                f"{self.base_url}/api/show",
                {"name": self.model},
            )
            details = body.get("details")
            detail_digest = details.get("digest") if isinstance(details, dict) else None
            digest = body.get("digest") or detail_digest or self.model
            if not isinstance(digest, str) or not digest:
                raise LLMBadJSONError("Ollama /api/show returned an invalid model digest")
            self._digest = digest
        return self._digest

    def generate(self, prompt: str, schema: dict[str, Any]) -> LLMResult:
        digest = self._ensure_digest()
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "format": schema,
            "stream": False,
            "keep_alive": KEEP_ALIVE,
            "options": {
                "temperature": self.temperature,
                "seed": self.seed,
                "num_ctx": self.num_ctx,
            },
        }
        started = time.perf_counter()
        body = _post_json(self._client, f"{self.base_url}/api/chat", payload)
        latency_ms = (time.perf_counter() - started) * 1000
        message = body.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise LLMBadJSONError("Ollama /api/chat returned no message content")
        return LLMResult(
            text=content,
            tokens_in=_as_int(body.get("prompt_eval_count")),
            tokens_out=_as_int(body.get("eval_count")),
            latency_ms=latency_ms,
            model=self.model,
            model_digest=digest,
        )


class OpenAICompatProvider:
    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        num_ctx: int = DEFAULT_NUM_CTX,
        temperature: float = DEFAULT_TEMPERATURE,
        seed: int = DEFAULT_SEED,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        client: httpx.Client | None = None,
    ) -> None:
        resolved_base_url = (base_url or os.environ.get("OPENAI_COMPAT_BASE_URL", "")).strip()
        resolved_api_key = (api_key or os.environ.get("OPENAI_COMPAT_API_KEY", "")).strip()
        if not resolved_base_url or not resolved_api_key:
            raise LLMMissingConfigError(
                "OpenAI-compatible provider requires OPENAI_COMPAT_BASE_URL and "
                "OPENAI_COMPAT_API_KEY"
            )
        self.model = model
        self.base_url = resolved_base_url.rstrip("/")
        self.api_key = resolved_api_key
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.seed = seed
        self.timeout = timeout
        self._client = client or httpx.Client(timeout=timeout)

    def settings(self) -> ProviderSettings:
        return ProviderSettings(
            model=self.model,
            model_digest=self.model,
            num_ctx=self.num_ctx,
            temperature=self.temperature,
            seed=self.seed,
            provider="openai-compat",
            base_url=self.base_url,
        )

    def generate(self, prompt: str, schema: dict[str, Any]) -> LLMResult:
        effective_prompt = (
            f"{prompt}\n\nReturn JSON matching this schema:\n"
            f"{json.dumps(schema, sort_keys=True, separators=(',', ':'))}"
        )
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": effective_prompt}],
            "temperature": self.temperature,
            "seed": self.seed,
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        started = time.perf_counter()
        body = _post_json(self._client, f"{self.base_url}/chat/completions", payload, headers)
        latency_ms = (time.perf_counter() - started) * 1000
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMBadJSONError("OpenAI-compatible endpoint returned no choices")
        first = choices[0]
        message = first.get("message") if isinstance(first, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise LLMBadJSONError("OpenAI-compatible endpoint returned no message content")
        usage = body.get("usage")
        if not isinstance(usage, dict):
            usage = {}
        return LLMResult(
            text=content,
            tokens_in=_as_int(usage.get("prompt_tokens")),
            tokens_out=_as_int(usage.get("completion_tokens")),
            latency_ms=latency_ms,
            model=self.model,
            model_digest=self.model,
        )


def _as_int(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int):
        raise LLMBadJSONError("LLM endpoint returned a non-integer token count")
    return value


def make_cache_key(
    *,
    model_digest: str,
    prompt: str,
    schema: dict[str, Any],
    num_ctx: int,
    temperature: float,
    seed: int,
    prompt_version: str,
    attempt: int,
) -> str:
    payload = {
        "model_digest": model_digest,
        "prompt": prompt,
        "schema": schema,
        "num_ctx": num_ctx,
        "temperature": temperature,
        "seed": seed,
        "prompt_version": prompt_version,
        "attempt": attempt,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class ResponseCache:
    def __init__(self, db_path: Path = DEFAULT_CACHE_DB) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS responses (
                    key TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    tokens_in INTEGER NOT NULL,
                    tokens_out INTEGER NOT NULL,
                    latency_ms REAL NOT NULL,
                    model TEXT NOT NULL,
                    model_digest TEXT NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def get(self, key: str) -> LLMResult | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT text, tokens_in, tokens_out, latency_ms, model, model_digest
                FROM responses WHERE key = ?
                """,
                (key,),
            ).fetchone()
        if row is None:
            return None
        return LLMResult(
            text=row[0],
            tokens_in=row[1],
            tokens_out=row[2],
            latency_ms=row[3],
            model=row[4],
            model_digest=row[5],
            cached=True,
        )

    def put(self, key: str, result: LLMResult) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO responses
                (key, text, tokens_in, tokens_out, latency_ms, model, model_digest)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    result.text,
                    result.tokens_in,
                    result.tokens_out,
                    result.latency_ms,
                    result.model,
                    result.model_digest,
                ),
            )


JsonSchema = dict[str, Any]
CacheKeyFn = Callable[..., str]

__all__ = [
    "DEFAULT_CACHE_DB",
    "LLMError",
    "LLMProvider",
    "LLMResult",
    "LLMBadJSONError",
    "LLMMissingConfigError",
    "LLMTimeoutError",
    "LLMUnreachableError",
    "OllamaProvider",
    "OpenAICompatProvider",
    "ProviderSettings",
    "ResponseCache",
    "make_cache_key",
]
