from __future__ import annotations

from pathlib import Path

from citriage.bench.b1 import B1System, is_prompt_truncated
from citriage.bench.metrics import score_evidence
from citriage.cases import Case
from citriage.labels import LabelName
from citriage.llm.provider import (
    DEFAULT_SEED,
    LLMProvider,
    LLMResult,
    ProviderSettings,
    ResponseCache,
)
from tests.helpers import bench_case, bench_evidence, bench_label

INVALID = '{"label":"bogus","culprit_file":null,"cited_lines":[2]}'
VALID = '{"label":"test_failure","culprit_file":"tests/test_x.py","cited_lines":[2]}'


class FakeProvider:
    def __init__(self, responses: list[tuple[str, int]]) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def generate(self, prompt: str, schema: dict[str, object]) -> LLMResult:
        self.prompts.append(prompt)
        text, tokens_in = self.responses.pop(0)
        return LLMResult(
            text=text,
            tokens_in=tokens_in,
            tokens_out=5,
            latency_ms=25.0,
            model="fake",
            model_digest="fake-digest",
        )


def _make_system(
    tmp_path: Path,
    responses: list[tuple[str, int]],
    *,
    use_cache: bool = True,
    raw_lines: list[str] | None = None,
) -> tuple[B1System, FakeProvider]:
    raw_root = tmp_path / "raw"
    log_path = raw_root / "o__r" / "1" / "job_100.log"
    log_path.parent.mkdir(parents=True)
    if raw_lines is None:
        raw_lines = [
            "setup line",
            "FAILED tests/test_x.py::test_a",
            "##[error]Process completed with exit code 1.",
            "Post job cleanup",
            "ignored cleanup line",
        ]
    log_path.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
    provider = FakeProvider(responses)
    settings = ProviderSettings(
        model="fake",
        model_digest="fake-digest",
        num_ctx=4096,
        temperature=0.0,
        seed=DEFAULT_SEED,
        provider="fake",
    )
    system = B1System(
        provider=provider,
        settings=settings,
        cache=ResponseCache(tmp_path / "cache.sqlite3"),
        raw_root=raw_root,
        use_cache=use_cache,
    )
    return system, provider


def test_schema_failure_then_retry_succeeds(tmp_path: Path) -> None:
    system, provider = _make_system(tmp_path, [(INVALID, 100), (VALID, 100)])

    prediction = system.predict(bench_case())

    assert prediction.label == LabelName.TEST_FAILURE
    assert prediction.culprit_file == "tests/test_x.py"
    assert prediction.invalid_output is True
    assert prediction.retry_count == 1
    assert prediction.prompt_version == "b1-v1"
    assert len(provider.prompts) == 2
    assert "previous response was invalid" in provider.prompts[1]


def test_retry_failure_falls_back_to_unknown(tmp_path: Path) -> None:
    system, _ = _make_system(tmp_path, [(INVALID, 100), (INVALID, 100)])

    prediction = system.predict(bench_case())

    assert prediction.label == LabelName.UNKNOWN
    assert prediction.culprit_file is None
    assert prediction.invalid_output is True
    assert prediction.retry_count == 1


def test_hallucinated_cited_line_is_detected(tmp_path: Path) -> None:
    raw_lines = [
        "setup line",
        "FAILED tests/test_x.py::test_a",
        "##[error]Process completed with exit code 1.",
        "Post job cleanup",
    ]
    prediction_text = '{"label":"test_failure","culprit_file":null,"cited_lines":[2,999]}'
    system, _ = _make_system(tmp_path, [(prediction_text, 100)], raw_lines=raw_lines)
    case = bench_case()

    prediction = system.predict(case)
    metrics = score_evidence(case, bench_label(evidence_lines=[2]), prediction, raw_lines)

    assert prediction.cited_lines == (2, 999)
    assert prediction.shown_line_numbers == (1, 2)
    assert metrics.hallucinated_citations == 1
    assert metrics.total_citations == 2
    assert metrics.citation_gold_hits == 1
    assert metrics.citation_gold_lines == 2


def test_cache_hit_avoids_live_provider_call(tmp_path: Path) -> None:
    system, provider = _make_system(tmp_path, [(VALID, 100)])
    case = bench_case()

    first = system.predict(case)
    second = system.predict(case)

    assert first.cached is False
    assert second.cached is True
    assert second.label == first.label
    assert len(provider.prompts) == 1


def test_no_cache_forces_live_calls(tmp_path: Path) -> None:
    system, provider = _make_system(tmp_path, [(VALID, 100), (VALID, 100)], use_cache=False)
    case = bench_case()

    first = system.predict(case)
    second = system.predict(case)

    assert first.cached is False
    assert second.cached is False
    assert len(provider.prompts) == 2


def test_retry_attempts_are_cached_separately(tmp_path: Path) -> None:
    system, provider = _make_system(tmp_path, [(INVALID, 100), (VALID, 100)])
    case = bench_case()

    first = system.predict(case)
    second = system.predict(case)

    assert first.retry_count == 1
    assert second.retry_count == 1
    assert second.label == LabelName.TEST_FAILURE
    assert second.cached is True
    assert len(provider.prompts) == 2


def test_prompt_uses_log_window_not_parser_evidence(tmp_path: Path) -> None:
    system, provider = _make_system(tmp_path, [(VALID, 100)])
    case: Case = bench_case(
        evidence=[bench_evidence(id="parser-only-id", text="PARSER ONLY EVIDENCE")]
    )

    system.predict(case)

    prompt = provider.prompts[0]
    assert "test_failure: Root cause" in prompt
    assert "2: FAILED tests/test_x.py::test_a" in prompt
    assert "parser-only-id" not in prompt
    assert "PARSER ONLY EVIDENCE" not in prompt


def test_prompt_truncation_heuristic() -> None:
    long_prompt = "x" * 400

    assert is_prompt_truncated(long_prompt, 1, 4096) is True
    assert is_prompt_truncated(long_prompt, 1000, 4096) is False
    assert is_prompt_truncated("short", 4096, 4096) is True


def test_fake_provider_satisfies_protocol() -> None:
    provider: LLMProvider = FakeProvider([(VALID, 100)])
    assert isinstance(provider.generate("prompt", {"type": "object"}), LLMResult)
