from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from citriage.labels import (
    Label,
    LabelName,
    append_label,
    base_labeler,
    latest_labels,
    load_labels,
)


def _label(**overrides: object) -> Label:
    data: dict[str, object] = {
        "repo": "pytest-dev/pytest",
        "run_id": 123,
        "job_id": 456,
        "label": LabelName.TEST_FAILURE,
        "labeler": "alice",
        "labeled_at": datetime(2026, 11, 19, 12, 0, tzinfo=UTC),
    }
    data.update(overrides)
    return Label.model_validate(data)


@pytest.mark.parametrize("name", list(LabelName))
def test_label_accepts_every_taxonomy_value(name: LabelName) -> None:
    label = _label(label=name)

    assert label.label is name


def test_label_rejects_unknown_taxonomy_value() -> None:
    with pytest.raises(ValidationError):
        _label(label="not_a_label")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("repo", "bad"),
        ("repo", "owner/"),
        ("run_id", 0),
        ("job_id", -1),
        ("evidence_lines", [0]),
        ("evidence_lines", [-2]),
    ],
)
def test_label_rejects_invalid_identity_and_evidence(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        _label(**{field: value})


@pytest.mark.parametrize(
    "culprit_file",
    [
        "/absolute/path.py",
        "\\absolute\\path.py",
        "src\\module.py",
        "../outside.py",
        "src/../outside.py",
        "src//module.py",
    ],
)
def test_label_rejects_non_repo_relative_culprit_file(culprit_file: str) -> None:
    with pytest.raises(ValidationError):
        _label(culprit_file=culprit_file)


def test_label_allows_null_and_valid_culprit_fields() -> None:
    label = _label(culprit_file=None, culprit_test=None)
    assert label.culprit_file is None
    assert label.culprit_test is None

    label = _label(culprit_file="src/pkg/module.py", culprit_test="tests/test_module.py::test_x")
    assert label.culprit_file == "src/pkg/module.py"
    assert label.culprit_test == "tests/test_module.py::test_x"


@pytest.mark.parametrize("value", ["", "   "])
def test_label_rejects_blank_labeler_and_culprit_test(value: str) -> None:
    with pytest.raises(ValidationError):
        _label(labeler=value)
    with pytest.raises(ValidationError):
        _label(culprit_test=value)


@pytest.mark.parametrize("flaky", [None, True, False])
def test_flaky_accepts_unknown_confirmed_and_real_failure(flaky: bool | None) -> None:
    assert _label(flaky=flaky).flaky is flaky


def test_append_and_load_roundtrip_preserves_null_flaky(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    label = _label(flaky=None, evidence_lines=[184, 185])

    append_label(label, path)

    assert load_labels(path) == [label]
    assert '"flaky":null' in path.read_text(encoding="utf-8")


def test_append_allows_duplicate_job_labels(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    first = _label(notes="first pass")
    second = _label(notes="second pass")

    append_label(first, path)
    append_label(second, path)

    assert load_labels(path) == [first, second]


def test_load_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_labels(tmp_path / "missing.jsonl") == []


def test_load_ignores_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    label = _label()
    path.write_text(
        "\n" + label.model_dump_json() + "\n\n",
        encoding="utf-8",
    )

    assert load_labels(path) == [label]


def test_load_rejects_invalid_line_with_line_number(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    path.write_text(
        _label().model_dump_json() + "\n" + "{not-json}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r":2: invalid label"):
        load_labels(path)


def test_latest_wins_and_undo_records_clear_latest_label(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    first = _label(case_id="c1", notes="first")
    second = _label(
        case_id="c1",
        notes="second",
        labeled_at=first.labeled_at + timedelta(seconds=1),
    )
    undo = Label(
        case_id="c1",
        repo=first.repo,
        run_id=first.run_id,
        job_id=first.job_id,
        action="undo",
        label=None,
        supersedes=second.id,
        labeler=first.labeler,
        labeled_at=second.labeled_at + timedelta(seconds=1),
    )

    append_label(first, path)
    append_label(second, path)

    assert latest_labels(load_labels(path))[("c1", "alice")].notes == "second"

    append_label(undo, path)

    assert len(load_labels(path)) == 3
    assert latest_labels(load_labels(path)) == {}


def test_label_action_requires_label_only_for_label_records() -> None:
    with pytest.raises(ValidationError):
        _label(action="label", label=None)
    with pytest.raises(ValidationError):
        _label(action="undo", label=LabelName.UNKNOWN)


def test_base_labeler_strips_second_round_suffix() -> None:
    assert base_labeler("alice-r2") == "alice"
    assert base_labeler("alice") == "alice"
