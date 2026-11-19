from citriage.parse.logparse import parse_log
from citriage.signature import normalize_for_signature, signature_from_log


def test_normalize_removes_digits_paths_hex_and_timestamps() -> None:
    raw = "2026-11-20T10:00:00Z src/pkg/mod.py:98 failed 0xdeadbeef 12345"
    assert normalize_for_signature(raw) == "<time> <path>:# failed <hex> #"


def test_signature_matches_similar_first_evidence() -> None:
    first = parse_log(
        "\n".join(
            [
                "##[group]Run pytest",
                "2026-11-20T10:00:00Z FAILED tests/a.py::test_x - AssertionError: got 42",
                "##[error]Process completed with exit code 1.",
                "##[group]Post job cleanup.",
            ]
        )
    )
    second = parse_log(
        "\n".join(
            [
                "##[group]Run pytest",
                "2026-11-20T11:00:00Z FAILED tests/b.py::test_x - AssertionError: got 99",
                "##[error]Process completed with exit code 1.",
                "##[group]Post job cleanup.",
            ]
        )
    )
    assert signature_from_log(first) == signature_from_log(second)


def test_empty_log_has_stable_signature() -> None:
    parsed = parse_log("")
    assert signature_from_log(parsed) == signature_from_log(parsed)
