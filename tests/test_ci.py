from pathlib import Path

import yaml

WORKFLOW = Path(".github/workflows/ci.yml")


def load_workflow() -> dict[str, object]:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def workflow_steps(workflow: dict[str, object]) -> list[dict[str, object]]:
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    test_job = jobs["test"]
    assert isinstance(test_job, dict)
    steps = test_job["steps"]
    assert isinstance(steps, list)
    return steps


def step_with_name(steps: list[dict[str, object]], name: str) -> dict[str, object]:
    return next(step for step in steps if step.get("name") == name)


def test_ci_workflow_uses_read_only_windows_job_and_required_actions() -> None:
    workflow = load_workflow()
    assert workflow["on"] == {"push": None, "pull_request": None}
    assert workflow["permissions"] == {"contents": "read"}

    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    test_job = jobs["test"]
    assert isinstance(test_job, dict)
    assert test_job["runs-on"] == "windows-latest"

    steps = workflow_steps(workflow)
    checkout = next(step for step in steps if step.get("uses") == "actions/checkout@v6")
    assert checkout["with"] == {"fetch-depth": 0}
    setup_uv = next(
        step
        for step in steps
        if step.get("uses")
        == "astral-sh/setup-uv@c771a70e6277c0a99b617c7a806ffedaca235ff9"
    )
    assert setup_uv["with"] == {"enable-cache": True, "version": "0.12.1"}
    setup_python = next(
        step for step in steps if step.get("uses") == "actions/setup-python@v6"
    )
    assert setup_python["with"] == {"python-version": "3.11"}


def test_ci_workflow_runs_the_local_quality_and_smoke_contract() -> None:
    steps = workflow_steps(load_workflow())

    assert step_with_name(steps, "Install")["run"] == (
        "uv sync --extra cpu --extra media --extra ml --extra tracking --group dev"
    )
    assert step_with_name(steps, "Lint")["run"] == "uv run ruff check src tests"
    assert step_with_name(steps, "Format")["run"] == (
        "uv run ruff format --check src tests"
    )
    assert step_with_name(steps, "Lock")["run"] == "uv lock --check"
    assert step_with_name(steps, "Documentation")["run"] == "uv run ddf-docs"
    assert step_with_name(steps, "Tests")["run"] == "uv run pytest"
    assert step_with_name(steps, "Tracked smoke")["run"] == (
        "uv run --extra tracking ddf run --root . --config configs/local.yaml "
        "--config configs/smoke.yaml"
    )
    assert step_with_name(steps, "Detector evaluator fixture smoke")["run"] == (
        "uv run pytest tests/test_detector_cli.py -k detector_compare_fixture_smoke"
    )


def test_ci_workflow_checks_pull_request_documentation_contract() -> None:
    step = step_with_name(
        workflow_steps(load_workflow()), "Pull request change contract"
    )
    assert step["if"] == "github.event_name == 'pull_request'"
    assert step["run"] == (
        "uv run ddf-docs --changed-from ${{ github.event.pull_request.base.sha }}"
    )
