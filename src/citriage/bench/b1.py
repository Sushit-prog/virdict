from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from citriage.bench.systems import (
    Prediction,
    register_system_factory,
)
from citriage.cases import Case
from citriage.labels import LabelName
from citriage.llm.provider import (
    DEFAULT_CACHE_DB,
    DEFAULT_NUM_CTX,
    DEFAULT_TIMEOUT_SECONDS,
    LLMProvider,
    LLMResult,
    OllamaProvider,
    ProviderSettings,
    ResponseCache,
    make_cache_key,
)
from citriage.parse.logparse import CLEANUP_PREFIXES, EXIT_CODE_RE, _normalize
from citriage.storage import log_path

PROMPT_VERSION = "b1-v1"
WINDOW_SIZE = 80
PROMPT_TRUNCATION_RATIO = 0.5

TAXONOMY_DEFINITIONS: tuple[tuple[LabelName, str], ...] = (
    (
        LabelName.TEST_FAILURE,
        "Root cause is an assertion, test error, or test collection failure.",
    ),
    (
        LabelName.LINT_FORMAT,
        "Root cause is a style, formatting, or static lint rule.",
    ),
    (
        LabelName.TYPE_ERROR,
        "Root cause is a static type-checking failure.",
    ),
    (
        LabelName.DEPENDENCY,
        "Root cause is dependency resolution, installation, or a missing/incompatible package.",
    ),
    (
        LabelName.INFRA_FLAKE,
        "Root cause is runner, network, or service infrastructure rather than repository code.",
    ),
    (
        LabelName.CONFIG_ERROR,
        "Root cause is invalid workflow, repository, or tool configuration.",
    ),
    (
        LabelName.UNKNOWN,
        "Evidence is insufficient or no other label fits; use as a last resort.",
    ),
)

B1_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": [label.value for label in LabelName]},
        "culprit_file": {"type": ["string", "null"]},
        "cited_lines": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["label", "culprit_file", "cited_lines"],
    "additionalProperties": False,
}


class B1Response(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: LabelName
    culprit_file: str | None
    cited_lines: list[int] = Field(default_factory=list)


def _is_cleanup(line: str) -> bool:
    return any(line.startswith(prefix) for prefix in CLEANUP_PREFIXES)


def build_log_window(raw_lines: list[str], limit: int = WINDOW_SIZE) -> tuple[tuple[int, str], ...]:
    normalized = [_normalize(line).strip() for line in raw_lines]
    cleanup_index = next(
        (index for index, line in enumerate(normalized) if _is_cleanup(line)),
        len(normalized),
    )
    anchor_index: int | None = None
    for index in range(cleanup_index - 1, -1, -1):
        if EXIT_CODE_RE.match(normalized[index]):
            anchor_index = index
            break
    end = anchor_index if anchor_index is not None else cleanup_index
    window: list[tuple[int, str]] = []
    for index in range(end - 1, -1, -1):
        line = normalized[index]
        if not line or _is_cleanup(line) or EXIT_CODE_RE.match(line):
            continue
        window.append((index + 1, line))
        if len(window) >= limit:
            break
    window.reverse()
    return tuple(window)


def build_prompt(case_id: str, window: tuple[tuple[int, str], ...]) -> str:
    lines = [
        "Classify the root cause of this failed GitHub Actions job.",
        f"Case: {case_id}",
        "",
        "Labels:",
    ]
    lines.extend(f"- {label.value}: {definition}" for label, definition in TAXONOMY_DEFINITIONS)
    lines.extend(
        [
            "",
            "Return JSON with exactly these keys: label, culprit_file, cited_lines.",
            "culprit_file must be a repository-relative POSIX path or null.",
            "cited_lines must contain only line numbers from the log window shown below.",
            "",
            "Log window (original line number: normalized line):",
        ]
    )
    lines.extend(f"{number}: {text}" for number, text in window)
    lines.extend(
        [
            "",
            "JSON response:",
        ]
    )
    return "\n".join(lines)


def estimate_prompt_tokens(prompt: str) -> int:
    return max(1, len(prompt) // 4)


def is_prompt_truncated(prompt: str, tokens_in: int, num_ctx: int) -> bool:
    estimated = estimate_prompt_tokens(prompt)
    return tokens_in >= num_ctx or tokens_in < estimated * PROMPT_TRUNCATION_RATIO


def _parse_response(text: str) -> B1Response | None:
    try:
        payload = json.loads(text)
        return B1Response.model_validate(payload)
    except (ValueError, ValidationError):
        return None


class B1System:
    prompt_version = PROMPT_VERSION

    def __init__(
        self,
        *,
        provider: LLMProvider,
        settings: ProviderSettings,
        cache: ResponseCache,
        raw_root: Path = Path("data/raw"),
        use_cache: bool = True,
        window_size: int = WINDOW_SIZE,
    ) -> None:
        self.provider = provider
        self.settings = settings
        self.cache = cache
        self.raw_root = raw_root
        self.use_cache = use_cache
        self.window_size = window_size
        self.model = settings.model
        self.model_digest = settings.model_digest
        self.num_ctx = settings.num_ctx

    def predict(self, case: Case) -> Prediction:
        path = log_path(
            self.raw_root,
            case.repo,
            case.run_id,
            case.job_id,
            run_dir_name=case.run_dir_name or None,
        )
        raw_lines: list[str] = []
        if path.exists():
            raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        window = build_log_window(raw_lines, self.window_size)
        base_prompt = build_prompt(case.case_id, window)

        parsed: B1Response | None = None
        result: LLMResult | None = None
        invalid_output = False
        retry_count = 0

        for attempt in (1, 2):
            prompt = base_prompt
            if attempt == 2:
                prompt += (
                    "\n\nYour previous response was invalid. Return only JSON matching the schema."
                )
            key = make_cache_key(
                model_digest=self.settings.model_digest,
                prompt=prompt,
                schema=B1_SCHEMA,
                num_ctx=self.settings.num_ctx,
                temperature=self.settings.temperature,
                seed=self.settings.seed,
                prompt_version=PROMPT_VERSION,
                attempt=attempt,
            )
            cached = self.cache.get(key) if self.use_cache else None
            llm_result = cached or self.provider.generate(prompt, B1_SCHEMA)
            if cached is None:
                self.cache.put(key, llm_result)
            result = llm_result
            parsed = _parse_response(llm_result.text)
            if parsed is not None:
                break
            invalid_output = True
            if attempt == 1:
                retry_count += 1

        if parsed is None:
            parsed = B1Response(label=LabelName.UNKNOWN, culprit_file=None, cited_lines=[])
        assert result is not None
        shown_lines = tuple(number for number, _ in window)
        return Prediction(
            label=parsed.label,
            culprit_file=parsed.culprit_file,
            cited_lines=tuple(parsed.cited_lines),
            shown_line_numbers=shown_lines,
            raw_tokens_in=result.tokens_in,
            raw_tokens_out=result.tokens_out,
            latency_ms=result.latency_ms,
            model=self.settings.model,
            model_digest=self.settings.model_digest,
            num_ctx=self.settings.num_ctx,
            prompt_version=PROMPT_VERSION,
            cached=result.cached,
            invalid_output=invalid_output,
            retry_count=retry_count,
            prompt_truncated=is_prompt_truncated(
                prompt,
                result.tokens_in,
                self.settings.num_ctx,
            ),
        )


def build_b1_system(
    *,
    model: str,
    raw_root: Path = Path("data/raw"),
    cache_db: Path = DEFAULT_CACHE_DB,
    num_ctx: int = DEFAULT_NUM_CTX,
    use_cache: bool = True,
    llm_timeout: float = DEFAULT_TIMEOUT_SECONDS,
    provider: LLMProvider | None = None,
    settings: ProviderSettings | None = None,
) -> B1System:
    resolved_provider = provider or OllamaProvider(
        model=model,
        num_ctx=num_ctx,
        timeout=llm_timeout,
    )
    resolved_settings = settings or resolved_provider.settings()  # type: ignore[attr-defined]
    return B1System(
        provider=resolved_provider,
        settings=resolved_settings,
        cache=ResponseCache(cache_db),
        raw_root=raw_root,
        use_cache=use_cache,
    )


def register_b1() -> None:
    register_system_factory("b1", build_b1_system, replace=True)


__all__ = [
    "B1_SCHEMA",
    "B1Response",
    "B1System",
    "PROMPT_VERSION",
    "TAXONOMY_DEFINITIONS",
    "build_b1_system",
    "build_log_window",
    "build_prompt",
    "estimate_prompt_tokens",
    "is_prompt_truncated",
    "register_b1",
]
