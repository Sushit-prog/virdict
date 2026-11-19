from __future__ import annotations

from citriage.redact import REDACTED, redact, redact_strings


def test_github_tokens_are_redacted() -> None:
    text = "a ghp_" + "a" * 36 + " b gho_" + "b" * 36 + " c github_pat_" + "c" * 40

    result = redact(text)

    assert "ghp_" not in result
    assert "gho_" not in result
    assert "github_pat_" not in result
    assert result.count(REDACTED) == 3


def test_bearer_tokens_are_redacted() -> None:
    result = redact("Authorization: Bearer abc.def-123_456")

    assert result == f"Authorization: Bearer {REDACTED}"


def test_aws_keys_are_redacted() -> None:
    text = (
        "key AKIA1234567890ABCDEF and ASIA1234567890ABCDEF "
        'aws_secret_access_key = "abcdefghijklmnopqrstuvwxyz1234567890ABCD"'
    )

    result = redact(text)

    assert "AKIA1234567890ABCDEF" not in result
    assert "ASIA1234567890ABCDEF" not in result
    assert "abcdefghijklmnopqrstuvwxyz1234567890ABCD" not in result
    assert result.count(REDACTED) == 3


def test_add_mask_values_are_removed_everywhere() -> None:
    text = "before\n::add-mask::s3cr3t-value\nafter s3cr3t-value and s3cr3t-value\n"

    result = redact(text)

    assert "s3cr3t-value" not in result
    assert f"::add-mask::{REDACTED}" in result


def test_extra_secret_is_redacted() -> None:
    result = redact("token=my-test-token", ("my-test-token",))

    assert result == f"token={REDACTED}"


def test_redaction_is_idempotent() -> None:
    text = "ghp_" + "a" * 36 + "\nBearer abc.def\n::add-mask::secret\nsecret\n"
    once = redact(text)

    assert redact(once, ("secret",)) == once


def test_redact_strings_recurses() -> None:
    value = {
        "workflow_name": "CI ghp_" + "a" * 36,
        "pull_requests": [
            {
                "base_ref": "AKIA1234567890ABCDEF",
                "nested": ["Bearer abc.def", 3, True, None],
            }
        ],
    }

    result = redact_strings(value)

    assert isinstance(result, dict)
    assert "ghp_" not in str(result)
    assert "AKIA1234567890ABCDEF" not in str(result)
    assert "abc.def" not in str(result)


def test_normal_text_is_unchanged() -> None:
    text = "tests/test_api.py::test_login PASSED\n"

    assert redact(text) == text
