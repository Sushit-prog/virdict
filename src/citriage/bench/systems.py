from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

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
    model: str = ""
    model_digest: str = ""
    num_ctx: int | None = None
    prompt_version: str = ""
    cached: bool = False
    invalid_output: bool = False
    retry_count: int = 0
    prompt_truncated: bool = False
    cited_lines: tuple[int, ...] = ()
    shown_line_numbers: tuple[int, ...] = ()


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
_FACTORIES: dict[str, Callable[..., System]] = {}


def register_system(name: str, system: System, *, replace: bool = False) -> None:
    if not name or (name in _SYSTEMS and not replace):
        raise ValueError(f"invalid or duplicate system name: {name!r}")
    _SYSTEMS[name] = system


def register_system_factory(
    name: str,
    factory: Callable[..., System],
    *,
    replace: bool = False,
) -> None:
    if not name or (name in _FACTORIES and not replace):
        raise ValueError(f"invalid or duplicate system factory: {name!r}")
    _FACTORIES[name] = factory


def get_system(name: str) -> System:
    try:
        return _SYSTEMS[name]
    except KeyError as exc:
        raise KeyError(f"unknown system {name!r}; available: {', '.join(system_names())}") from exc


def build_system(name: str, **options: Any) -> System:
    if name in _FACTORIES:
        return _FACTORIES[name](**options)
    return get_system(name)


def system_names() -> tuple[str, ...]:
    return tuple(sorted(set(_SYSTEMS) | set(_FACTORIES)))


register_system("b0", B0System())


__all__ = [
    "B0System",
    "Prediction",
    "System",
    "build_system",
    "get_system",
    "register_system",
    "register_system_factory",
    "system_names",
]
