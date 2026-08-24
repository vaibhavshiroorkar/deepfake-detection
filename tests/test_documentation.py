from pathlib import Path

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


def test_cli_reference_discovers_every_leaf_command() -> None:
    assert command_paths(build_parser()) == (
        "ddf cache build",
        "ddf evaluate",
        "ddf features export",
        "ddf features score",
        "ddf manifest build",
        "ddf predict",
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
