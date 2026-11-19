from pathlib import Path

from citriage.baselines.b0 import predict
from citriage.labels import LabelName
from citriage.parse.logparse import Evidence, ParsedLog, parse_log

FIXTURES = Path(__file__).parent / "fixtures" / "logs"


def test_b0_labels_and_culprits() -> None:
    expectations = {
        "pytest_failure.log": (LabelName.TEST_FAILURE, "tests/test_api.py"),
        "ruff.log": (LabelName.LINT_FORMAT, "src/foo.py"),
        "mypy.log": (LabelName.TYPE_ERROR, "src/pkg/mod.py"),
        "uv_resolve.log": (LabelName.DEPENDENCY, None),
        "timeout.log": (LabelName.INFRA_FLAKE, None),
        "error_only.log": (LabelName.CONFIG_ERROR, None),
    }
    for name, (label, culprit) in expectations.items():
        parsed = parse_log((FIXTURES / name).read_text(encoding="utf-8"))
        result = predict(parsed)
        assert result.label == label, name
        assert result.culprit_file == culprit, name
        assert result.evidence_ids, name


def test_b0_prefers_in_region_evidence() -> None:
    early = Evidence(
        id="early",
        kind="dependency",
        line_no=2,
        text="ERROR: Could not find a version that satisfies the requirement foo",
        tool="pip",
        in_region=False,
    )
    late = Evidence(
        id="late",
        kind="pytest_summary",
        line_no=20,
        text="FAILED tests/test_api.py::test_create - AssertionError: expected 201",
        tool="pytest",
        in_region=True,
    )
    parsed = ParsedLog(10, 25, [early, late], ["pytest"], False)
    result = predict(parsed)
    assert result.label == LabelName.TEST_FAILURE
    assert result.evidence_ids == ["late"]
    assert result.culprit_file == "tests/test_api.py"


def test_b0_falls_back_to_whole_log_without_in_region_evidence() -> None:
    item = Evidence(
        id="only",
        kind="infra",
        line_no=2,
        text="The operation was canceled.",
        tool="infra",
        in_region=False,
    )
    result = predict(ParsedLog(10, 25, [item], [], False))
    assert result.label == LabelName.INFRA_FLAKE
    assert result.evidence_ids == ["only"]


def test_b0_no_evidence_is_unknown() -> None:
    result = predict(ParsedLog(1, 0, [], [], False))
    assert result.label == LabelName.UNKNOWN
    assert result.culprit_file is None
    assert result.evidence_ids == []
