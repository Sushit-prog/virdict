from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from citriage.bench.b1 import B1System, build_b1_system
from citriage.bench.systems import B0System, build_system, get_system, system_names
from citriage.labels import LabelName
from citriage.llm.provider import DEFAULT_SEED, ProviderSettings
from tests.helpers import bench_case, bench_evidence


def test_prediction_defaults_and_forbids_extra_fields() -> None:
    from citriage.bench.systems import Prediction

    prediction = Prediction(label=LabelName.UNKNOWN)
    assert prediction.culprit_file is None
    assert prediction.cited_evidence_ids == []
    assert prediction.raw_tokens_in == 0
    assert prediction.raw_tokens_out == 0
    assert prediction.latency_ms == 0.0

    with pytest.raises(ValidationError):
        Prediction(label=LabelName.UNKNOWN, unexpected=1)


def test_b0_is_registered() -> None:
    assert "b0" in system_names()
    assert isinstance(get_system("b0"), B0System)


def test_b1_factory_is_registered(tmp_path: Path) -> None:
    assert "b1" in system_names()

    class _Provider:
        def generate(self, prompt: str, schema: dict[str, object]) -> object:
            raise AssertionError("not called")

    settings = ProviderSettings(
        model="fake",
        model_digest="fake-digest",
        num_ctx=4096,
        temperature=0.0,
        seed=DEFAULT_SEED,
        provider="fake",
    )
    system = build_b1_system(
        model="fake",
        cache_db=tmp_path / "cache.sqlite3",
        provider=_Provider(),  # type: ignore[arg-type]
        settings=settings,
    )

    assert isinstance(system, B1System)
    assert build_system("b0").__class__ is B0System


def test_unknown_system_lists_available_systems() -> None:
    with pytest.raises(KeyError, match="available: b0"):
        get_system("does-not-exist")


def test_b0_adapter_recomputes_from_case_evidence() -> None:
    evidence = bench_evidence(kind="pytest_summary", text="FAILED tests/test_x.py::test_a")
    case = bench_case(evidence=[evidence]).model_copy(
        update={"b0_label": LabelName.UNKNOWN, "b0_evidence_ids": []}
    )

    prediction = B0System().predict(case)

    assert prediction.label == LabelName.TEST_FAILURE
    assert prediction.cited_evidence_ids == [evidence.id]
