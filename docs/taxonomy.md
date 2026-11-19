# Failure taxonomy

Apply one primary label per failed job. Label the first root-cause failure, not downstream
noise. If a later job fails because an earlier job failed, label the earlier failure and use the
later job only as supporting evidence.

`flaky` is not a taxonomy label. It is a separate boolean-like field:

- `null`: unknown; no conclusion was recorded.
- `true`: a later green run of the same commit confirms the flake.
- `false`: the labeler judged this to be a real failure, not a flake.

`culprit_file` is a repository-relative POSIX path or `null`. Use `null` when the failure cannot
be attributed to a specific file. When a label needs a test identifier, use `culprit_test`.

## test_failure

The root cause is an assertion, test error, or test collection failure. The fix would change
test code, application code exercised by a test, or test fixtures.

Example log signatures:

- `FAILED tests/test_api.py::test_create - AssertionError: expected 201, got 500`
- `E   assert 3 == 4` followed by a pytest failure summary

## lint_format

The root cause is a style, formatting, or static lint rule. The fix would change formatting or
lint-clean code, not behavior or types.

Example log signatures:

- `ruff check .` followed by `Found 3 errors.`
- `would reformat path/to/file.py` from a format check

## type_error

The root cause is a static type-checking failure. The fix would change type annotations,
stubs, overloads, or type-compatible code.

Example log signatures:

- `error: Incompatible return value type (got "str", expected "int")`
- `pyright` output containing `error: Type "None" is not assignable to type "str"`

## dependency

The root cause is dependency resolution, installation, or a missing or incompatible package.
The fix would change dependency declarations, lock files, indexes, or version constraints.

Example log signatures:

- `ERROR: Could not find a version that satisfies the requirement package==9.9.9`
- `ResolutionImpossible` or `Because package depends on ... version solving failed`

## infra_flake

The root cause is runner, network, or service infrastructure rather than the repository's
code. The fix would retry, increase timeouts, or change CI infrastructure. Do not use this label
for a deterministic test failure that happens to run in CI.

Example log signatures:

- `The runner has received a shutdown signal. This can happen when the runner service is stopped.`
- `429 Too Many Requests` or `Timeout waiting for a service to become healthy`

## config_error

The root cause is invalid workflow, repository, or tool configuration. The fix would change a
YAML workflow, required input, secret reference, matrix entry, or path configuration.

Example log signatures:

- `Invalid workflow file` with a YAML or expression error
- `Input required and not supplied: token` or a missing-secret configuration error

## unknown

The evidence is insufficient, the log is too truncated, or no other label fits. Prefer a more
specific label when the root cause is visible. Use `unknown` as a last resort, not as a synonym
for "not investigated".

Example log signatures:

- Only `Process completed with exit code 1.` with no preceding diagnostic
- A truncated log whose first retained line is already a downstream failure

## Tie-break rules

- Label the first root-cause failure, not a later aggregate or gate job.
- If two labels fit, choose the label for the change the fix would touch.
- A dependency failure that is only a transient network timeout is `infra_flake`; a resolution
  or version conflict is `dependency`.
- A test failure caused by a type error is `type_error`; a test failure caused by an assertion is
  `test_failure`.
- A formatting failure reported by a test command is `lint_format`; a failing assertion is
  `test_failure`.
- A workflow that cannot start because of its own configuration is `config_error`, not
  `infra_flake`.
- If the first root cause cannot be identified from the retained log, use `unknown` and record
  the evidence lines that show why.
