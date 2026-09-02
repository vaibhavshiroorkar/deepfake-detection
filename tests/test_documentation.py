import shlex
from pathlib import Path

import pytest

from deepfake_detection.cli import build_parser
from deepfake_detection.documentation.checks import (
    check_change_contract,
    check_cli_reference,
    check_external_links,
    check_markdown_tree,
)
from deepfake_detection.documentation.cli_reference import (
    command_paths,
    render_command_block,
)

SMOKE_RECIPE_DOCUMENTS = ("README.md", "docs/reproducibility.md")


def command_tokens(command: str) -> tuple[str, ...]:
    return tuple(shlex.split(command))


def extra_values(tokens: tuple[str, ...]) -> set[str]:
    return {
        tokens[index + 1]
        for index, argument in enumerate(tokens[:-1])
        if argument == "--extra"
    }


def smoke_recipe_commands(relative: str) -> tuple[tuple[str, ...], ...]:
    text = Path(relative).read_text(encoding="utf-8")
    smoke_block = next(
        block
        for block in text.split("```powershell")[1:]
        if "configs/smoke.yaml" in block
    ).split("```", maxsplit=1)[0]
    return tuple(
        command_tokens(line.strip())
        for line in smoke_block.splitlines()
        if line.strip().startswith("uv ")
    )


def test_published_smoke_recipes_install_and_run_with_media_and_tracking() -> None:
    required_extras = {"media", "tracking"}
    for relative in SMOKE_RECIPE_DOCUMENTS:
        commands = smoke_recipe_commands(relative)
        install = next(command for command in commands if command[:2] == ("uv", "sync"))
        run = next(
            command
            for command in commands
            if command[:2] == ("uv", "run")
            and "ddf" in command
            and "run" in command
            and "configs/smoke.yaml" in command
        )
        assert required_extras <= extra_values(install)
        assert required_extras <= extra_values(run)
        for command in commands:
            if "mlflow" in command and "server" in command:
                assert required_extras <= extra_values(command)


def assert_markdown_headings(
    root: Path, relative: str, headings: tuple[str, ...]
) -> None:
    text = (root / relative).read_text(encoding="utf-8")
    for heading in headings:
        assert f"## {heading}\n" in text


FOUNDATION_CHAPTERS = {
    "docs/handbook/00-learning-path.md": (
        "Who this is for",
        "Reading order",
        "How to study each chapter",
        "Learning checks",
    ),
    "docs/handbook/01-problem-and-research-question.md": (
        "Problem definition",
        "Research questions",
        "Contribution",
        "Limits",
        "Viva questions",
        "Sources",
    ),
    "docs/handbook/02-deep-learning-foundations.md": (
        "Tensors and shapes",
        "Forward pass and gradients",
        "Binary classification loss",
        "Optimization and regularization",
        "Transfer learning",
        "Worked example",
        "Exercises",
        "Viva questions",
        "Sources",
    ),
    "docs/handbook/03-audio-video-foundations.md": (
        "Digital video",
        "Digital audio",
        "Timestamps and synchronization",
        "Codecs and shortcuts",
        "Worked timeline",
        "Exercises",
        "Viva questions",
        "Sources",
    ),
    "docs/handbook/04-data-and-leakage.md": (
        "Manifest contract",
        "Cue-specific labels",
        "Source-disjoint splits",
        "Leakage and shortcuts",
        "Method holdout",
        "Project code path",
        "Failure cases",
        "Exercises",
        "Viva questions",
        "Sources",
    ),
}


@pytest.mark.parametrize(("relative", "headings"), FOUNDATION_CHAPTERS.items())
def test_foundation_chapter_contracts(relative: str, headings: tuple[str, ...]) -> None:
    assert_markdown_headings(Path.cwd(), relative, headings)


PREPROCESSING_HEADINGS = (
    "Pipeline overview",
    "Shared timeline",
    "Media decoding",
    "Face detection and tracking",
    "Visual and mouth crops",
    "Audio views",
    "Quality gates and abstention",
    "Caching and hashes",
    "Current limitations",
    "Project code path",
    "Failure cases",
    "Exercises",
    "Viva questions",
    "Sources",
)


def test_preprocessing_chapter_contract() -> None:
    assert_markdown_headings(
        Path.cwd(),
        "docs/handbook/05-preprocessing-pipeline.md",
        PREPROCESSING_HEADINGS,
    )


MODEL_CHAPTERS = {
    "docs/handbook/06-visual-branch.md": (
        "Cue and hypothesis",
        "Input shape",
        "Architecture",
        "Forward pass",
        "Training target",
        "Candidate comparison",
        "Current limitations",
        "Project code path",
        "Exercises",
        "Viva questions",
        "Sources",
    ),
    "docs/handbook/07-audio-branch.md": (
        "Cue and hypothesis",
        "Waveform representation",
        "Architecture",
        "Attention pooling",
        "Training target",
        "Padding and masks",
        "Candidate comparison",
        "Current limitations",
        "Project code path",
        "Exercises",
        "Viva questions",
        "Sources",
    ),
    "docs/handbook/08-sync-branch.md": (
        "Cue and hypothesis",
        "Correspondence task",
        "Offset classes",
        "Architecture",
        "Losses",
        "Negative pairs",
        "Candidate comparison",
        "Current limitations",
        "Project code path",
        "Exercises",
        "Viva questions",
        "Sources",
    ),
    "docs/handbook/09-fusion-and-calibration.md": (
        "Why late fusion",
        "Out-of-fold predictions",
        "Calibration",
        "Fusion features",
        "Missing evidence",
        "Ablations",
        "Current limitations",
        "Project code path",
        "Exercises",
        "Viva questions",
        "Sources",
    ),
}


@pytest.mark.parametrize(("relative", "headings"), MODEL_CHAPTERS.items())
def test_model_chapter_contracts(relative: str, headings: tuple[str, ...]) -> None:
    assert_markdown_headings(Path.cwd(), relative, headings)


def test_cli_reference_discovers_every_leaf_command() -> None:
    assert command_paths(build_parser()) == (
        "ddf cache build",
        "ddf detector compare",
        "ddf detector fetch-yunet",
        "ddf detector run",
        "ddf detector sample",
        "ddf detector validate-annotations",
        "ddf evaluate",
        "ddf features export",
        "ddf features score",
        "ddf manifest build",
        "ddf predict",
        "ddf run",
        "ddf smoke",
        "ddf split build",
        "ddf split crossfit",
        "ddf split method-holdout",
        "ddf threshold",
        "ddf train audio",
        "ddf train fusion",
        "ddf train sync",
        "ddf train visual",
    )


def write_cli_reference(root: Path, block: str) -> None:
    path = root / "docs" / "reference" / "cli.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        "# CLI\n\n<!-- BEGIN GENERATED COMMANDS -->\n"
        f"{block}\n"
        "<!-- END GENERATED COMMANDS -->\n",
        encoding="utf-8",
    )


def test_cli_reference_accepts_the_current_parser(tmp_path: Path) -> None:
    write_cli_reference(tmp_path, render_command_block(build_parser()))

    assert check_cli_reference(tmp_path) == ()


def test_cli_reference_rejects_a_stale_command_block(tmp_path: Path) -> None:
    write_cli_reference(tmp_path, "- `ddf stale`")

    assert check_cli_reference(tmp_path)[0].rule == "cli-reference"


def test_markdown_checker_accepts_valid_ascii_documentation(tmp_path: Path) -> None:
    guide = tmp_path / "docs" / "guide.md"
    guide.parent.mkdir()
    guide.write_text("# Guide\n\nRead [the root](../README.md).\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Project\n", encoding="utf-8")

    assert check_markdown_tree(tmp_path) == ()


def test_markdown_checker_ignores_external_dataset_documentation(
    tmp_path: Path,
) -> None:
    readme = tmp_path / "data" / "external" / "README.md"
    readme.parent.mkdir(parents=True)
    readme.write_text("External\u2014text.\n", encoding="utf-8")

    assert check_markdown_tree(tmp_path) == ()


def test_markdown_checker_reports_broken_links_and_prose_rules(
    tmp_path: Path,
) -> None:
    guide = tmp_path / "guide.md"
    guide.write_text(
        "# Guide\n\n[Missing](missing.md)\n\n[INCOMPLETE]\n\nBad\u2014dash.\n",
        encoding="utf-8",
    )

    rules = {issue.rule for issue in check_markdown_tree(tmp_path)}

    assert rules == {"ascii", "local-link", "unfinished-marker"}


def test_markdown_checker_ignores_inline_and_fenced_code(tmp_path: Path) -> None:
    guide = tmp_path / "guide.md"
    guide.write_text(
        """# Guide

`[Inline](missing-inline.md) [INCOMPLETE]`

```markdown
[Triple](missing-triple.md)
[INCOMPLETE]
```

````markdown
```markdown
[Longer](missing-longer.md)
[INCOMPLETE]
```
````

~~~markdown
[Tilde](missing-tilde.md)
[INCOMPLETE]
~~~
""",
        encoding="utf-8",
    )

    assert check_markdown_tree(tmp_path) == ()


def test_change_contract_requires_docs_and_changelog_for_source_changes() -> None:
    issues = check_change_contract(
        (Path("src/deepfake_detection/cli.py"), Path("tests/test_cli.py"))
    )

    assert {issue.rule for issue in issues} == {
        "code-needs-documentation",
        "material-change-needs-changelog",
    }


def test_change_contract_accepts_code_docs_and_changelog_together() -> None:
    assert (
        check_change_contract(
            (
                Path("src/deepfake_detection/cli.py"),
                Path("tests/test_cli.py"),
                Path("docs/reference/cli.md"),
                Path("CHANGELOG.md"),
            )
        )
        == ()
    )


def test_detector_annotation_protocol_defines_visible_face_error_accounting() -> None:
    protocol = " ".join(
        Path("docs/model-selection.md").read_text(encoding="utf-8").split()
    )

    for required in (
        "box for every visible face",
        "at most one suitable speaking target",
        "five landmarks",
        "Unmatched detections are false detections",
        "other visible faces are not false positives",
    ):
        assert required in protocol


def test_external_link_checker_reports_only_failed_targets(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "[Good](https://example.com/good) [Bad](https://example.com/bad)\n",
        encoding="utf-8",
    )
    statuses = {
        "https://example.com/bad": 404,
        "https://example.com/good": 200,
    }

    issues = check_external_links(tmp_path, statuses.__getitem__)

    assert len(issues) == 1
    assert "https://example.com/bad" in issues[0].message
