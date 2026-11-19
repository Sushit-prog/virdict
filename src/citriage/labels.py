from __future__ import annotations

import json
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

LABELS_PATH = Path("data/labels/labels.jsonl")


class LabelName(StrEnum):
    TEST_FAILURE = "test_failure"
    LINT_FORMAT = "lint_format"
    TYPE_ERROR = "type_error"
    DEPENDENCY = "dependency"
    INFRA_FLAKE = "infra_flake"
    CONFIG_ERROR = "config_error"
    UNKNOWN = "unknown"


class Label(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repo: str = Field(pattern=r"^[^/\s]+/[^/\s]+$")
    run_id: int = Field(gt=0)
    job_id: int = Field(gt=0)
    label: LabelName
    flaky: bool | None = None
    culprit_file: str | None = None
    culprit_test: str | None = None
    evidence_lines: list[Annotated[int, Field(gt=0)]] = Field(default_factory=list)
    notes: str = ""
    labeler: str = Field(min_length=1)
    labeled_at: datetime

    @field_validator("culprit_file")
    @classmethod
    def _validate_culprit_file(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value or value.startswith(("/", "\\")) or "\\" in value:
            raise ValueError("culprit_file must be a relative POSIX repo path")
        parts = value.split("/")
        if any(part in {"", ".", ".."} for part in parts):
            raise ValueError("culprit_file must not contain empty, '.' or '..' components")
        return value

    @field_validator("culprit_test")
    @classmethod
    def _validate_culprit_test(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.strip():
            raise ValueError("culprit_test must not be empty")
        return value

    @field_validator("labeler")
    @classmethod
    def _validate_labeler(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("labeler must not be empty")
        return value


def append_label(label: Label, path: Path = LABELS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(label.model_dump_json() + "\n")


def load_labels(path: Path = LABELS_PATH) -> list[Label]:
    if not path.exists():
        return []
    labels: list[Label] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                labels.append(Label.model_validate(json.loads(line)))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid label: {exc}") from exc
    return labels


__all__ = [
    "LABELS_PATH",
    "Label",
    "LabelName",
    "append_label",
    "load_labels",
]
