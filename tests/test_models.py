from citriage.models import WorkflowRun


def test_workflow_run_reads_api_path_field() -> None:
    run = WorkflowRun.model_validate(
        {
            "id": 1,
            "name": "CI",
            "path": ".github/workflows/ci.yml",
            "created_at": "2026-01-01T00:00:00Z",
            "html_url": "https://github.com/o/r/actions/runs/1",
        }
    )

    assert run.workflow_path == ".github/workflows/ci.yml"


def test_workflow_run_still_reads_workflow_path_field() -> None:
    run = WorkflowRun.model_validate(
        {
            "id": 1,
            "name": "CI",
            "workflow_path": ".github/workflows/ci.yml",
            "created_at": "2026-01-01T00:00:00Z",
            "html_url": "https://github.com/o/r/actions/runs/1",
        }
    )

    assert run.workflow_path == ".github/workflows/ci.yml"
