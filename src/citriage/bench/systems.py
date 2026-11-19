from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from citriage.baselines.b0 import predict as predict_b0
from citriage.cases import Case
from citriage.labels import LabelName
from citriage.parse.logparse import ParsedLog


class Prediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: LabelName
    culprit_file: str | None = None
    cited_evidence_ids: list[str] = Field(default_factory=list)
    raw_tokens_in: int = 0
    raw_tokens_out: int = 0
    latency_ms: float = 0.0


class System(Protocol):
    def predict(self, case: Case) -> Prediction: ...


class B0System:
    def predict(self, case: Case) -> Prediction:
        parsed = ParsedLog(
            region_start=case.region_start,
            region_end=case.region_end,
            evidence=case.evidence,
            tools_detected=case.tools_detected,
            has_nontrivial_error=case.has_nontrivial_error,
        )
        result = predict_b0(parsed)
        return Prediction(
            label=result.label,
            culprit_file=result.culprit_file,
            cited_evidence_ids=list(result.evidence_ids),
        )


_SYSTEMS: dict[str, System] = {}


def register_system(name: str, system: System) -> None:
    if not name or name in _SYSTEMS:
        raise ValueError(f"invalid or duplicate system name: {name!r}")
    _SYSTEMS[name] = system


def get_system(name: str) -> System:
    try:
        return _SYSTEMS[name]
    except KeyError as exc:
        raise KeyError(f"unknown system {name!r}; available: {', '.join(system_names())}") from exc


def system_names() -> tuple[str, ...]:
    return tuple(sorted(_SYSTEMS))


register_system("b0", B0System())


__all__ = [
    "B0System",
    "Prediction",
    "System",
    "get_system",
    "register_system",
    "system_names",
]
