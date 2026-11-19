# Label schema

Labels are stored in `data/labels/labels.jsonl`, one JSON object per line. The file is committed
to the repository. `data/raw/` remains ignored and is not part of the label store.

The store is append-only. Appending a second label for the same `repo`, `run_id`, and `job_id` is
allowed so relabeling history is preserved. Consumers should treat the last record for a job as
the current label when a single label is needed.

## Fields

| Field | Type | Required | Rules |
| --- | --- | --- | --- |
| `repo` | string | yes | `owner/name` with no whitespace. |
| `run_id` | integer | yes | Positive GitHub Actions run ID. |
| `job_id` | integer | yes | Positive GitHub Actions job ID. |
| `label` | enum | yes | One of the taxonomy values below. |
| `flaky` | boolean or null | no | `null` means unknown, `true` means a later green run of the same commit confirmed the flake, `false` means the labeler judged it a real failure. Defaults to `null`. |
| `culprit_file` | string or null | no | Repository-relative POSIX path, or `null` when unknown. Absolute paths, backslashes, empty components, `.` and `..` are rejected. |
| `culprit_test` | string or null | no | Test identifier or other test-level culprit, or `null`. An empty or whitespace-only value is rejected. |
| `evidence_lines` | list of integers | no | 1-based line numbers in the stored log file. Each number must be positive. Defaults to an empty list. |
| `notes` | string | no | Free-form notes. Defaults to an empty string. |
| `labeler` | string | yes | Non-empty identifier for the labeler. |
| `labeled_at` | string | yes | ISO 8601 timestamp. |

Unknown fields are rejected so misspelled fields do not silently enter the store.

## Taxonomy values

- `test_failure`
- `lint_format`
- `type_error`
- `dependency`
- `infra_flake`
- `config_error`
- `unknown`

See `docs/taxonomy.md` for definitions, example log signatures, and tie-break rules.

## Flaky semantics

`flaky` is independent of the primary label:

- `null` records that no flake conclusion was made.
- `true` requires evidence of a later green run of the same commit. The flakes collector writes
  snapshots with `flake_confirmed=true` and `later_green_same_sha=true` in `meta.json`.
- `false` records a labeler's judgment that the failure is real. It is not proof that no retry
  could ever pass.

## Evidence lines

`evidence_lines` refer to the stored `job_<id>.log` file, not the original GitHub log. Logs may
have their head dropped during collection, so line numbers are always relative to the retained
file. Use 1-based line numbers.

## Example

```json
{"repo":"pytest-dev/pytest","run_id":123456,"job_id":789012,"label":"test_failure","flaky":false,"culprit_file":"src/_pytest/python.py","culprit_test":"testing/test_python.py::test_raises","evidence_lines":[184,185,186],"notes":"Assertion mismatch in fixture cleanup.","labeler":"alice","labeled_at":"2026-11-19T12:00:00Z"}
```

## Validation

The `citriage.labels.Label` model validates every field before a record is appended or loaded.
`load_labels()` reports the line number when a JSONL record is malformed. Empty lines are ignored.
