from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from citriage.llm.provider import (
    DEFAULT_SEED,
    LLMBadJSONError,
    LLMResult,
    LLMTimeoutError,
    OllamaProvider,
    OpenAICompatProvider,
    ResponseCache,
    make_cache_key,
)

SCHEMA = {"type": "object", "properties": {"label": {"type": "string"}}}


@respx.mock
def test_ollama_uses_schema_options_and_fetches_digest_once() -> None:
    show_route = respx.post("http://localhost:11434/api/show").mock(
        return_value=httpx.Response(200, json={"details": {"digest": "sha256:abc"}})
    )
    chat_route = respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={
                "message": {"content": "{}"},
                "prompt_eval_count": 12,
                "eval_count": 3,
            },
        )
    )
    provider = OllamaProvider("qwen", num_ctx=2048)

    first = provider.generate("prompt one", SCHEMA)
    provider.generate("prompt two", SCHEMA)

    assert show_route.call_count == 1
    assert chat_route.call_count == 2
    assert first.tokens_in == 12
    assert first.tokens_out == 3
    assert first.model_digest == "sha256:abc"
    payload = json.loads(chat_route.calls[0].request.content)
    assert payload["format"] == SCHEMA
    assert payload["keep_alive"] == "30s"
    assert payload["options"] == {
        "temperature": 0.0,
        "seed": DEFAULT_SEED,
        "num_ctx": 2048,
    }


@respx.mock
def test_ollama_digest_falls_back_to_model_name() -> None:
    respx.post("http://localhost:11434/api/show").mock(return_value=httpx.Response(200, json={}))
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(200, json={"message": {"content": "{}"}})
    )

    result = OllamaProvider("qwen").generate("prompt", SCHEMA)

    assert result.model_digest == "qwen"


@respx.mock
def test_ollama_timeout_is_typed() -> None:
    respx.post("http://localhost:11434/api/show").mock(
        return_value=httpx.Response(200, json={"digest": "sha256:abc"})
    )
    respx.post("http://localhost:11434/api/chat").mock(side_effect=httpx.ConnectTimeout("slow"))

    with pytest.raises(LLMTimeoutError):
        OllamaProvider("qwen").generate("prompt", SCHEMA)


@respx.mock
def test_ollama_bad_json_is_typed() -> None:
    respx.post("http://localhost:11434/api/show").mock(
        return_value=httpx.Response(200, text="oops")
    )

    with pytest.raises(LLMBadJSONError):
        OllamaProvider("qwen").generate("prompt", SCHEMA)


@respx.mock
def test_openai_compat_uses_generic_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "https://llm.example/v1/")
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "secret")
    route = respx.post("https://llm.example/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "{}"}}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 2},
            },
        )
    )

    result = OpenAICompatProvider("compat-model").generate("prompt", SCHEMA)

    assert result.tokens_in == 8
    assert result.tokens_out == 2
    request = route.calls[0].request
    assert request.headers["authorization"] == "Bearer secret"
    payload = json.loads(request.content)
    assert payload["response_format"] == {"type": "json_object"}
    assert "prompt" in payload["messages"][0]["content"]


def test_cache_key_changes_with_num_ctx() -> None:
    common = {
        "model_digest": "digest",
        "prompt": "prompt",
        "schema": SCHEMA,
        "temperature": 0.0,
        "seed": DEFAULT_SEED,
        "prompt_version": "b1-v1",
        "attempt": 1,
    }

    first = make_cache_key(**common, num_ctx=4096)
    second = make_cache_key(**common, num_ctx=2048)

    assert first != second


def test_response_cache_hit_preserves_latency_and_tokens(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "cache.sqlite3")
    key = make_cache_key(
        model_digest="digest",
        prompt="prompt",
        schema=SCHEMA,
        num_ctx=4096,
        temperature=0.0,
        seed=DEFAULT_SEED,
        prompt_version="b1-v1",
        attempt=1,
    )
    original = LLMResult(
        text="{}",
        tokens_in=10,
        tokens_out=4,
        latency_ms=123.5,
        model="model",
        model_digest="digest",
    )

    assert cache.get(key) is None
    cache.put(key, original)
    cached = cache.get(key)

    assert cached is not None
    assert cached.cached is True
    assert cached.tokens_in == 10
    assert cached.tokens_out == 4
    assert cached.latency_ms == 123.5
