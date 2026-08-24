# Documentation Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the learning handbook, technical references, research evidence files, decision records, and automated documentation checks for the implemented platform.

**Architecture:** Markdown remains the readable source of truth. A small Python documentation package validates local links, punctuation, unfinished markers, CLI drift, and change coverage. Tests define required chapter contracts, while `docs/README.md` maps each source package to its teaching and reference owners.

**Tech Stack:** Python 3.11 through 3.13, pathlib, argparse, subprocess, pytest, Markdown, Mermaid, GitHub Actions, uv, Ruff

**Spec:** `docs/superpowers/specs/2026-08-24-project-handbook-and-research-evidence-design.md`

## Global Constraints

- Work directly on `main` and preserve a clean commit after each task.
- Assume the reader knows basic Python and Git only.
- Teach PyTorch, deep learning, statistics, and audio-video processing from first principles.
- Use plain English and the repository's ASCII punctuation rules.
- Use primary papers, official documentation, dataset papers, and source repositories.
- Mark future behavior as planned. Describe only implemented behavior as current.
- Do not invent experiment results, run IDs, benchmark scores, or findings.
- Do not store raw videos, face crops, private paths, or restricted artifacts.
- Use stable file paths and symbol names in repository documentation.
- Keep `README.md` short. Put teaching material under `docs/handbook/`.
- Put operational details under `docs/reference/`.
- Put experiment definitions and accepted evidence under `docs/research/`.
- Update code, tests, related documentation, and `CHANGELOG.md` in the same commit.
- No empty handbook chapter counts as implementation.
- Run `uv run ruff check src tests`, `uv run ruff format --check src tests`, `uv lock --check`, `uv run ddf-docs`, and `uv run pytest` before final completion.

Every technical handbook chapter must include learning goals, required
background, theory, defined equations, tensor shapes, a worked example, the
project code path, design trade-offs, failure cases, supporting tests,
exercises, viva questions, and primary sources when those sections apply.

## Program decomposition

This plan implements the documentation foundation and documents the current
platform. The remaining program stays ordered in `ROADMAP.md`. Create a new
focused implementation plan at each later gate:

1. Local MLflow and versioned configuration.
2. Landmark-aware face views and the detector benchmark.
3. Audio masks and strong branch baselines.
4. Out-of-fold fusion, calibration, and validation experiments.
5. Locked evaluation, findings, paper, model card, and viva package.

Later plans must use the interfaces that actually exist after the preceding
gate. They must not guess signatures in advance.

---

### Task 1: Repository-owned documentation checker

**Files:**
- Create: `src/deepfake_detection/documentation/__init__.py`
- Create: `src/deepfake_detection/documentation/checks.py`
- Create: `src/deepfake_detection/documentation/__main__.py`
- Create: `tests/test_documentation.py`
- Modify: `pyproject.toml`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: a repository root `Path` and an optional Git base reference.
- Produces: `DocumentationIssue`, `check_markdown_tree()`,
  `check_external_links()`, `check_change_contract()`, `git_changed_paths()`,
  and the `ddf-docs` command.

- [ ] **Step 1: Write failing unit tests for Markdown validation**

Add these contracts to `tests/test_documentation.py`:

```python
from pathlib import Path

from deepfake_detection.documentation.checks import (
    check_change_contract,
    check_external_links,
    check_markdown_tree,
)


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
        "# Guide\n\n[Missing](missing.md)\n\n[IN" "COMPLETE]\n\nBad\u2014dash.\n",
        encoding="utf-8",
    )

    rules = {issue.rule for issue in check_markdown_tree(tmp_path)}

    assert rules == {"ascii", "local-link", "unfinished-marker"}


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
```

- [ ] **Step 2: Run the documentation tests and confirm the missing package failure**

Run:

```powershell
uv run pytest tests\test_documentation.py -v
```

Expected: collection fails because `deepfake_detection.documentation` does not
exist.

- [ ] **Step 3: Implement the validation model and pure checks**

Create `src/deepfake_detection/documentation/checks.py` with these exact public
types and functions:

```python
from __future__ import annotations

import re
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import unquote
from urllib.request import Request, urlopen


@dataclass(frozen=True, slots=True, order=True)
class DocumentationIssue:
    path: Path
    rule: str
    message: str


IGNORED_PARTS = frozenset({".git", ".pytest_cache", ".venv"})
INCOMPLETE_MARKERS = ("[IN" + "COMPLETE]", "PENDING" + " CONTENT")
FORBIDDEN_CHARACTERS = frozenset(
    {"\u2013", "\u2014", "\u2022", "\u00b7", "\u2026", "\u2192"}
)
LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
FENCED_CODE_PATTERN = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_PATTERN = re.compile(r"`[^`\n]+`")


def _markdown_files(root: Path) -> tuple[Path, ...]:
    return tuple(
        path
        for path in sorted(root.rglob("*.md"))
        if not IGNORED_PARTS.intersection(path.relative_to(root).parts)
    )


def check_markdown_tree(root: Path) -> tuple[DocumentationIssue, ...]:
    issues: list[DocumentationIssue] = []
    for path in _markdown_files(root):
        text = path.read_text(encoding="utf-8")
        prose = INLINE_CODE_PATTERN.sub("", FENCED_CODE_PATTERN.sub("", text))
        relative = path.relative_to(root)
        if any(character in text for character in FORBIDDEN_CHARACTERS):
            issues.append(
                DocumentationIssue(relative, "ascii", "Forbidden Unicode punctuation")
            )
        if any(marker in prose for marker in INCOMPLETE_MARKERS):
            issues.append(
                DocumentationIssue(
                    relative,
                    "unfinished-marker",
                    "Unfinished documentation marker",
                )
            )
        for match in LINK_PATTERN.finditer(prose):
            target = match.group(1).strip().strip("<>")
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            local_target = unquote(target.split("#", 1)[0])
            if local_target and not (path.parent / local_target).exists():
                issues.append(
                    DocumentationIssue(
                        relative,
                        "local-link",
                        f"Missing local target: {target}",
                    )
                )
    return tuple(sorted(issues))


def check_change_contract(
    changed_paths: Sequence[Path],
) -> tuple[DocumentationIssue, ...]:
    paths = tuple(changed_paths)
    source_changed = any(path.parts[:1] == ("src",) for path in paths)
    teaching_docs_changed = any(
        path.suffix == ".md" and path.name != "CHANGELOG.md" for path in paths
    )
    material_changed = source_changed or any(
        path.as_posix()
        in {
            "docs/data-card.md",
            "docs/model-selection.md",
            "docs/research-design.md",
            "docs/reproducibility.md",
            "docs/threat-model.md",
        }
        for path in paths
    )
    changelog_changed = Path("CHANGELOG.md") in paths
    issues: list[DocumentationIssue] = []
    if source_changed and not teaching_docs_changed:
        issues.append(
            DocumentationIssue(
                Path("docs"),
                "code-needs-documentation",
                "Source changes require a related Markdown update",
            )
        )
    if material_changed and not changelog_changed:
        issues.append(
            DocumentationIssue(
                Path("CHANGELOG.md"),
                "material-change-needs-changelog",
                "Material changes require a changelog entry",
            )
        )
    return tuple(sorted(issues))


def git_changed_paths(root: Path, base_ref: str) -> tuple[Path, ...]:
    result = subprocess.run(
        ["git", "-C", str(root), "diff", "--name-only", f"{base_ref}...HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(Path(line) for line in result.stdout.splitlines() if line)


def external_urls(root: Path) -> tuple[str, ...]:
    urls: set[str] = set()
    for path in _markdown_files(root):
        text = path.read_text(encoding="utf-8")
        prose = INLINE_CODE_PATTERN.sub("", FENCED_CODE_PATTERN.sub("", text))
        for match in LINK_PATTERN.finditer(prose):
            target = match.group(1).strip().strip("<>")
            if target.startswith(("http://", "https://")):
                urls.add(target)
    return tuple(sorted(urls))


def _fetch_status(url: str) -> int:
    request = Request(
        url,
        headers={"User-Agent": "deepfake-generalization-doc-checker"},
    )
    try:
        with urlopen(request, timeout=15) as response:  # noqa: S310
            return int(response.status)
    except HTTPError as error:
        return int(error.code)
    except URLError:
        return 0


def check_external_links(
    root: Path,
    fetch_status: Callable[[str], int] = _fetch_status,
) -> tuple[DocumentationIssue, ...]:
    issues = []
    for url in external_urls(root):
        status = fetch_status(url)
        if status == 0 or status >= 400:
            issues.append(
                DocumentationIssue(
                    Path("docs"),
                    "external-link",
                    f"External target returned status {status}: {url}",
                )
            )
    return tuple(sorted(issues))
```

Export the public symbols from `documentation/__init__.py`:

```python
from .checks import (
    DocumentationIssue,
    check_change_contract,
    check_external_links,
    check_markdown_tree,
    external_urls,
    git_changed_paths,
)

__all__ = [
    "DocumentationIssue",
    "check_change_contract",
    "check_external_links",
    "check_markdown_tree",
    "external_urls",
    "git_changed_paths",
]
```

- [ ] **Step 4: Implement the command entry point**

Create `src/deepfake_detection/documentation/__main__.py`:

```python
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .checks import (
    check_change_contract,
    check_external_links,
    check_markdown_tree,
    git_changed_paths,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ddf-docs")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--changed-from")
    parser.add_argument("--external", action="store_true")
    arguments = parser.parse_args(argv)
    root = arguments.root.resolve()
    issues = list(check_markdown_tree(root))
    if arguments.external:
        issues.extend(check_external_links(root))
    if arguments.changed_from:
        changed = git_changed_paths(root, arguments.changed_from)
        issues.extend(check_change_contract(changed))
    for issue in sorted(set(issues)):
        print(f"{issue.path}: {issue.rule}: {issue.message}")
    return int(bool(issues))


if __name__ == "__main__":
    raise SystemExit(main())
```

Add this script to `[project.scripts]` in `pyproject.toml`:

```toml
ddf-docs = "deepfake_detection.documentation.__main__:main"
```

- [ ] **Step 5: Run focused tests and the checker**

Run:

```powershell
uv run pytest tests\test_documentation.py -v
uv run ddf-docs
```

Expected: all focused tests pass and the checker exits with code zero.

- [ ] **Step 6: Record and commit the checker**

Add an `Unreleased` changelog entry for repository-owned documentation
validation. Add `uv run ddf-docs` to the README test commands so this source
change has a teaching-document update. Then run:

```powershell
git add pyproject.toml uv.lock README.md CHANGELOG.md src/deepfake_detection/documentation tests/test_documentation.py
git commit -m "Add documentation validation"
```

### Task 2: Public CLI contract and documentation index

**Files:**
- Create: `src/deepfake_detection/documentation/cli_reference.py`
- Create: `docs/README.md`
- Create: `docs/reference/cli.md`
- Modify: `src/deepfake_detection/cli.py:879`
- Modify: `src/deepfake_detection/documentation/checks.py`
- Modify: `src/deepfake_detection/documentation/__main__.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_documentation.py`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: the argparse parser returned by `build_parser()`.
- Produces: `build_parser()`, `command_paths()`, `render_command_block()`, and CLI reference drift validation.

- [ ] **Step 1: Write failing tests for the public parser and command tree**

Add to `tests/test_cli.py`:

```python
from deepfake_detection.cli import build_parser


def test_public_parser_exposes_the_documented_command_tree() -> None:
    parser = build_parser()
    assert parser.prog == "ddf"
```

Add to `tests/test_documentation.py`:

```python
from deepfake_detection.cli import build_parser
from deepfake_detection.documentation.cli_reference import command_paths


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
```

- [ ] **Step 2: Run the focused tests and confirm the missing interfaces**

Run:

```powershell
uv run pytest tests\test_cli.py tests\test_documentation.py -v
```

Expected: import failures for `build_parser` and `cli_reference`.

- [ ] **Step 3: Expose the parser and render leaf command paths**

Rename `_parser()` to `build_parser()` in `cli.py`. Change `main()` to call
`build_parser()`.

Create `documentation/cli_reference.py`:

```python
from __future__ import annotations

import argparse


def command_paths(
    parser: argparse.ArgumentParser,
    prefix: tuple[str, ...] = ("ddf",),
) -> tuple[str, ...]:
    child_paths: list[str] = []
    for action in parser._actions:
        choices = getattr(action, "choices", None)
        if not isinstance(choices, dict):
            continue
        for name, child in choices.items():
            child_paths.extend(command_paths(child, (*prefix, str(name))))
    if child_paths:
        return tuple(sorted(child_paths))
    return (" ".join(prefix),)


def render_command_block(parser: argparse.ArgumentParser) -> str:
    return "\n".join(f"- `{path}`" for path in command_paths(parser))
```

- [ ] **Step 4: Create the live documentation index and CLI reference**

Create `docs/README.md` with these sections:

- `# Documentation`
- `## Start here`
- `## Learning handbook`
- `## Technical reference`
- `## Research evidence`
- `## Decisions and project controls`
- `## Documentation ownership`
- `## Update rules`

Link only files that exist at this task. List future chapter paths in code
format until their tasks create them.

The ownership table must include these exact package mappings:

| Package | Handbook owner | Reference owner |
|---|---|---|
| `data` | `04-data-and-leakage.md` | `artifact-contracts.md` |
| `views` | `05-preprocessing-pipeline.md` | `architecture.md` |
| `branches` | `06-visual-branch.md` through `08-sync-branch.md` | `architecture.md` |
| `training` | `10-training-system.md` | `configuration.md` |
| `fusion` | `09-fusion-and-calibration.md` | `artifact-contracts.md` |
| `evaluation` | `11-evaluation-and-statistics.md` | `testing.md` |
| `inference` | `12-inference-and-dashboard.md` | `architecture.md` |
| `dashboard` | `12-inference-and-dashboard.md` | `architecture.md` |
| `documentation` | `00-learning-path.md` | `testing.md` |

Create `docs/reference/cli.md` with:

- Installation prerequisites.
- The PowerShell invocation pattern `uv run ddf`.
- A generated command section between `<!-- BEGIN GENERATED COMMANDS -->` and
  `<!-- END GENERATED COMMANDS -->`.
- A short description and `--help` command for all 14 leaf commands.
- The meaning of exit codes zero, one, and two where the current CLI uses them.
- A warning that joblib files can execute code during loading.

- [ ] **Step 5: Add CLI drift checking**

Add `check_cli_reference(root: Path) -> tuple[DocumentationIssue, ...]` to
`checks.py`. It must extract the generated block from `docs/reference/cli.md`
and compare it with `render_command_block(build_parser())`:

```python
CLI_BLOCK_START = "<!-- BEGIN GENERATED COMMANDS -->"
CLI_BLOCK_END = "<!-- END GENERATED COMMANDS -->"


def check_cli_reference(root: Path) -> tuple[DocumentationIssue, ...]:
    from deepfake_detection.cli import build_parser
    from deepfake_detection.documentation.cli_reference import render_command_block

    path = root / "docs" / "reference" / "cli.md"
    if not path.exists():
        return (
            DocumentationIssue(
                Path("docs/reference/cli.md"),
                "cli-reference",
                "CLI reference is missing",
            ),
        )
    text = path.read_text(encoding="utf-8")
    if CLI_BLOCK_START not in text or CLI_BLOCK_END not in text:
        return (
            DocumentationIssue(
                path.relative_to(root),
                "cli-reference",
                "Generated command markers are missing",
            ),
        )
    actual = text.split(CLI_BLOCK_START, 1)[1].split(CLI_BLOCK_END, 1)[0].strip()
    expected = render_command_block(build_parser())
    if actual != expected:
        return (
            DocumentationIssue(
                path.relative_to(root),
                "cli-reference",
                "Generated command block is stale",
            ),
        )
    return ()
```

Call `check_cli_reference(root)` from `documentation.__main__.main()`. Add these
tests:

```python
from deepfake_detection.documentation.checks import check_cli_reference
from deepfake_detection.documentation.cli_reference import render_command_block


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
```

- [ ] **Step 6: Run focused and full checks**

Run:

```powershell
uv run pytest tests\test_cli.py tests\test_documentation.py -v
uv run ddf-docs
uv run ddf --help
```

Expected: tests pass, the checker exits zero, and the help lists eight
top-level commands.

- [ ] **Step 7: Update the changelog and commit**

```powershell
git add CHANGELOG.md docs/README.md docs/reference/cli.md src/deepfake_detection/cli.py src/deepfake_detection/documentation tests/test_cli.py tests/test_documentation.py
git commit -m "Document the command and documentation contracts"
```

### Task 3: Learning foundations chapters

**Files:**
- Create: `docs/handbook/README.md`
- Create: `docs/handbook/00-learning-path.md`
- Create: `docs/handbook/01-problem-and-research-question.md`
- Create: `docs/handbook/02-deep-learning-foundations.md`
- Create: `docs/handbook/03-audio-video-foundations.md`
- Create: `docs/handbook/04-data-and-leakage.md`
- Modify: `docs/README.md`
- Modify: `tests/test_documentation.py`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: the approved design, current research design, data card, manifest, protocols, and timeline code.
- Produces: a linked beginner learning path through the problem, ML foundations, media foundations, and data protocol.

- [ ] **Step 1: Write the failing chapter contract test**

Add this helper and mapping to `tests/test_documentation.py`:

```python
import pytest


def assert_markdown_headings(root: Path, relative: str, headings: tuple[str, ...]) -> None:
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
```

- [ ] **Step 2: Run the contract test and confirm the missing files**

Run:

```powershell
uv run pytest tests\test_documentation.py::test_foundation_chapter_contracts -v
```

Expected: five failures because the chapters do not exist.

- [ ] **Step 3: Write the learning path and research problem chapters**

`00-learning-path.md` must define a 15-chapter reading sequence, a practice
cycle, expected time per chapter, fixture-based exercises, and progress checks.
It must direct readers to run tests after reading each implementation chapter.

`01-problem-and-research-question.md` must explain:

- Real media, face swaps, reenactment, synthetic speech, and combined attacks.
- Why benchmark accuracy can hide identity, codec, silence, and method leakage.
- RQ1 through RQ5 from the approved specification.
- The null and alternative hypothesis for fusion.
- Why the contribution is controlled evidence, not a new fusion architecture.
- Intended use, non-goals, and the meaning of a negative finding.

- [ ] **Step 4: Write the deep learning foundations chapter**

Use hand-calculated examples with a batch of two samples. Explain `[batch,
time, channels, height, width]` and `[batch, samples]` tensors. Derive binary
cross entropy with logits, gradient descent, weight decay, early stopping,
freezing, and unfreezing.

Map the theory to `BranchOutput`, `BinaryTrainingConfig`,
`run_accumulated_epoch()`, and `fit_binary_branch()`. State that automatic
mixed precision is planned, not current.

- [ ] **Step 5: Write the audio-video foundations chapter**

Use the current defaults to calculate:

- 16 visual frames per clip.
- 16,000 audio samples per second.
- 64,000 samples in the four-second audio view.
- 50 mouth frames in a two-second sync view at 25 FPS.
- A maximum offset of 0.32 seconds, equal to 5,120 samples.

Explain presentation timestamps, decoding, frame sampling, window overlap,
resampling, leading silence, codec artifacts, and why one shared timeline is
required.

- [ ] **Step 6: Write the data and leakage chapter**

Document every `ClipRecord` field and the four manipulation types. Explain why
visual uses `video_fake`, audio uses `audio_fake`, and fusion uses `clip_fake`.

Walk through `load_manifest()`, `build_source_split()`, `audit_split()`,
`identity_strict_subset()`, `split_hash()`, and
`build_method_holdout_protocol()`. Include a three-person hand-worked split
example and show why row-level random splitting is invalid.

- [ ] **Step 7: Link chapters and run validation**

Create `docs/handbook/README.md` with the five live links and the future reading
order in code format. Add the live links to `docs/README.md`.

Run:

```powershell
uv run pytest tests\test_documentation.py -v
uv run ddf-docs
```

Expected: all documentation tests and checks pass.

- [ ] **Step 8: Update the changelog and commit**

```powershell
git add CHANGELOG.md docs/README.md docs/handbook tests/test_documentation.py
git commit -m "Teach the project foundations"
```

### Task 4: Preprocessing pipeline chapter

**Files:**
- Create: `docs/handbook/05-preprocessing-pipeline.md`
- Modify: `docs/handbook/README.md`
- Modify: `docs/README.md`
- Modify: `tests/test_documentation.py`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: `ClipRecord`, `ViewConfig`, `FFmpegMediaDecoder`, `MTCNNFaceDetector`, tracking contracts, `Preprocessor`, `CacheStore`, and quality reports.
- Produces: a complete trace from one manifest row to cached visual, audio, and sync views.

- [ ] **Step 1: Add the failing preprocessing chapter contract**

Require these headings in `tests/test_documentation.py`:

```python
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
```

- [ ] **Step 2: Run the test and confirm the missing chapter**

```powershell
uv run pytest tests\test_documentation.py::test_preprocessing_chapter_contract -v
```

Expected: failure because the chapter does not exist.

- [ ] **Step 3: Write the pipeline and data-shape walkthrough**

Trace these calls in order:

```text
ClipRecord
  -> FFmpegMediaDecoder.inspect and decode methods
  -> ViewConfig and shared timestamps
  -> MTCNNFaceDetector.detect
  -> select_primary_track
  -> visual, audio, mouth, and sync views
  -> QualityReport.full_fusion_blockers
  -> preprocessing_config_hash and cache_fingerprint
  -> CacheStore.save
```

Document exact output shapes from the current defaults. Explain normalization,
padding, trimming, square expansion, leading-silence removal, real-context sync
offsets, and cache namespaces.

- [ ] **Step 4: Document current weaknesses without hiding them**

State these current limits explicitly:

- MTCNN is the only implemented detector.
- Its available landmarks are discarded by the adapter.
- The mouth crop uses the lower 48 percent of the face box.
- Greedy IoU tracking can switch identities.
- Multi-person clips may abstain.
- The planned YuNet and landmark work has no result yet.

Connect every weakness to its existing test or its planned detector experiment.

- [ ] **Step 5: Link, validate, and commit**

```powershell
uv run pytest tests\test_preprocessor.py tests\test_tracking.py tests\test_face_detector.py tests\test_documentation.py -v
uv run ddf-docs
git add CHANGELOG.md docs/README.md docs/handbook tests/test_documentation.py
git commit -m "Document preprocessing and view integrity"
```

### Task 5: Model branch and fusion chapters

**Files:**
- Create: `docs/handbook/06-visual-branch.md`
- Create: `docs/handbook/07-audio-branch.md`
- Create: `docs/handbook/08-sync-branch.md`
- Create: `docs/handbook/09-fusion-and-calibration.md`
- Modify: `docs/handbook/README.md`
- Modify: `docs/README.md`
- Modify: `tests/test_documentation.py`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: branch contracts, branch implementations, sync objectives, feature storage, calibration, and late fusion.
- Produces: theory and code walkthroughs for every current model path and planned candidate comparison.

- [ ] **Step 1: Add failing contracts for the four model chapters**

Add a parametrized mapping requiring these common headings:

```python
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
```

- [ ] **Step 2: Run the contracts and confirm four missing files**

```powershell
uv run pytest tests\test_documentation.py -k model_chapter -v
```

Expected: four failures for missing chapters.

- [ ] **Step 3: Write the visual chapter**

Explain spatial artifacts, ImageNet transfer learning, EfficientNet-B0 feature
extraction, per-frame embeddings, GRU temporal aggregation, and clip logits.
Trace `VisualArtifactBranch.forward()` and `build_efficientnet_b0()` with exact
tensor axes. Compare the current baseline with planned ConvNeXt-Tiny. Mark
DINOv2 as optional and unimplemented.

- [ ] **Step 4: Write the audio chapter**

Explain raw waveforms, Wav2Vec2 temporal tokens, projection, learned attention
weights, weighted pooling, and clip logits. Work through a three-token attention
example by hand.

State that padded batches currently lack valid-length attention masks. Explain
why this can leak duration. Compare Wav2Vec2 with planned WavLM and AASIST.

- [ ] **Step 5: Write the synchronization chapter**

Explain aligned pairs, shifted same-clip pairs, cross-identity mismatches,
offset classes at zero and both signs of 80, 160, and 320 milliseconds, and the
contrastive term.

Trace `SynchronizationBranch`, `crop_audio_context()`,
`contrastive_alignment_loss()`, `sync_anomaly_logit()`, and
`sync_training_loss()`. Explain why temporal tokens must remain unpooled before
alignment. Mark SyncNet-style and AV-HuBERT comparisons as planned.

- [ ] **Step 6: Write the fusion and calibration chapter**

Explain source-grouped cross-fitting with a four-source worked example. Explain
why in-sample branch predictions leak training knowledge into fusion.

Trace `FeatureRecord`, `FeatureStore.assemble()`, `LateFusion.fit()`,
`FusionArtifact`, and scoring. Derive Platt scaling and logistic fusion. Explain
quality features, missing branch rejection, MLP ablation, and threshold
separation.

- [ ] **Step 7: Link, validate, and commit**

```powershell
uv run pytest tests\test_branches.py tests\test_sync_objective.py tests\test_fusion.py tests\test_documentation.py -v
uv run ddf-docs
git add CHANGELOG.md docs/README.md docs/handbook tests/test_documentation.py
git commit -m "Document branch models and fusion"
```

### Task 6: Training system chapter

**Files:**
- Create: `docs/handbook/10-training-system.md`
- Modify: `docs/handbook/README.md`
- Modify: `docs/README.md`
- Modify: `tests/test_documentation.py`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: cached datasets, collation, training engines, stages, checkpoint contracts, provenance, and training CLI commands.
- Produces: a reproducible explanation of the current training lifecycle and its missing tracking features.

- [ ] **Step 1: Add the failing training chapter contract**

Require these headings:

```python
TRAINING_HEADINGS = (
    "Training data flow",
    "Datasets and batches",
    "Loss and optimization",
    "Gradient accumulation",
    "Freezing stages",
    "Early stopping",
    "Seeds and determinism",
    "Checkpoints and provenance",
    "Experiment tracking",
    "Hardware policy",
    "Project code path",
    "Failure cases",
    "Exercises",
    "Viva questions",
    "Sources",
)


def test_training_chapter_contract() -> None:
    assert_markdown_headings(
        Path.cwd(),
        "docs/handbook/10-training-system.md",
        TRAINING_HEADINGS,
    )
```

- [ ] **Step 2: Run the contract and confirm the missing chapter**

```powershell
uv run pytest tests\test_documentation.py::test_training_chapter_contract -v
```

Expected: failure because `10-training-system.md` does not exist.

- [ ] **Step 3: Write the training lifecycle**

Document `CachedBranchDataset`, `CachedSyncDataset`, collators, cue labels,
batch shapes, AdamW use in the CLI, binary and sync loss calls, accumulation,
freeze stages, early stopping, best-state restoration, and history JSON.

Use a five-batch, four-step accumulation example. Show that the final partial
group still produces one correctly scaled optimizer step.

- [ ] **Step 4: Explain provenance and current tracking status**

Document `RunMetadata`, `hash_config()`, `save_checkpoint()`,
`load_checkpoint()`, `validate_branch_states()`, and SHA-256 artifacts.

State that JSON histories and checkpoints are current. State that local MLflow,
automatic mixed precision, and versioned configuration files are planned.
Record the verified 16 GB VRAM constraint without claiming measured batch sizes.

- [ ] **Step 5: Link, validate, and commit**

```powershell
uv run pytest tests\test_training.py tests\test_training_recipes.py tests\test_checkpoint.py tests\test_documentation.py -v
uv run ddf-docs
git add CHANGELOG.md docs/README.md docs/handbook tests/test_documentation.py
git commit -m "Document the training lifecycle"
```

### Task 7: Evaluation, inference, reproduction, and viva chapters

**Files:**
- Create: `docs/handbook/11-evaluation-and-statistics.md`
- Create: `docs/handbook/12-inference-and-dashboard.md`
- Create: `docs/handbook/13-reproducing-the-project.md`
- Create: `docs/handbook/14-viva-preparation.md`
- Modify: `docs/handbook/README.md`
- Modify: `docs/README.md`
- Modify: `tests/test_documentation.py`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: metric functions, bootstrap functions, corruption functions, prediction engine, dashboard view model, README commands, and project risks.
- Produces: the final four chapters and a complete handbook reading path.

- [ ] **Step 1: Add failing contracts for the final chapters**

Require these exact headings:

```python
FINAL_CHAPTERS = {
    "docs/handbook/11-evaluation-and-statistics.md": (
        "Evaluation questions",
        "Confusion matrix metrics",
        "Ranking metrics",
        "Calibration metrics",
        "Threshold selection",
        "Coverage and abstention",
        "Bootstrap confidence intervals",
        "Paired comparison",
        "Subgroups and small samples",
        "Stress tests",
        "Project code path",
        "Exercises",
        "Viva questions",
        "Sources",
    ),
    "docs/handbook/12-inference-and-dashboard.md": (
        "Inference data flow",
        "Artifact loading",
        "One-video prediction",
        "Coverage gate",
        "Dashboard presentation",
        "Security and misuse",
        "Project code path",
        "Failure cases",
        "Exercises",
        "Viva questions",
    ),
    "docs/handbook/13-reproducing-the-project.md": (
        "Reproduction levels",
        "Environment setup",
        "Data preparation",
        "Smoke workflow",
        "Result workflow",
        "Artifact verification",
        "Troubleshooting",
        "Reproduction checklist",
    ),
    "docs/handbook/14-viva-preparation.md": (
        "Project in two minutes",
        "Research contribution",
        "Architecture questions",
        "Data and leakage questions",
        "Model questions",
        "Statistics questions",
        "Limitations and criticism",
        "Demonstration plan",
        "Questions the project cannot answer",
    ),
}


@pytest.mark.parametrize(("relative", "headings"), FINAL_CHAPTERS.items())
def test_final_chapter_contracts(relative: str, headings: tuple[str, ...]) -> None:
    assert_markdown_headings(Path.cwd(), relative, headings)
```

- [ ] **Step 2: Run the contracts and confirm four missing files**

```powershell
uv run pytest tests\test_documentation.py -k final_chapter -v
```

Expected: four failures for missing chapters.

- [ ] **Step 3: Write evaluation and statistics from first principles**

Build a hand-worked confusion matrix. Define every reported metric and show
when it fails. Explain ROC-AUC versus PR-AUC under class imbalance. Calculate a
small Brier score and expected calibration error example.

Trace `binary_metrics()`, `select_balanced_accuracy_threshold()`,
`evaluate_items()`, `per_method_metrics()`, `subgroup_metrics()`,
`cluster_bootstrap_interval()`, `paired_auc_difference()`, and corruptions.
Explain why source identities, not rows, are bootstrap units.

- [ ] **Step 4: Write inference and dashboard behavior**

Trace `InferenceConfig`, `load_prediction_engine()`, `PredictionEngine`,
`PredictionResult`, `build_view_model()`, and the Streamlit app. Explain the
coverage gate before the verdict. Include the joblib loading risk and safe
result language from the threat model.

- [ ] **Step 5: Write the reproducibility chapter**

Give PowerShell commands for environment setup, tests, CLI discovery, manifest
creation, splits, caching, branch training, cross-fitting, fusion, threshold
selection, evaluation, and inference. Label full-dataset commands as requiring
licensed data and trained artifacts.

Separate smoke reproduction, result reproduction, and artifact verification.
Do not claim that a full smoke configuration or MLflow command exists yet.

- [ ] **Step 6: Write the viva chapter**

Provide concise answers for at least 40 questions. Cover research novelty,
data leakage, cue labels, preprocessing, architectures, losses, fusion,
calibration, bootstrap intervals, external validity, subgroup limits, compute,
security, and negative findings.

Include a five-minute demonstration order and a list of claims the project
cannot support.

- [ ] **Step 7: Complete navigation, validate, and commit**

Link all 15 chapters from both `docs/handbook/README.md` and `docs/README.md`.

```powershell
uv run pytest tests\test_metrics.py tests\test_bootstrap.py tests\test_inference.py tests\test_dashboard_view.py tests\test_documentation.py -v
uv run ddf-docs
git add CHANGELOG.md docs/README.md docs/handbook tests/test_documentation.py
git commit -m "Complete the project learning handbook"
```

### Task 8: Technical reference set

**Files:**
- Create: `docs/reference/architecture.md`
- Create: `docs/reference/configuration.md`
- Create: `docs/reference/artifact-contracts.md`
- Create: `docs/reference/testing.md`
- Create: `docs/reference/hardware-and-compute.md`
- Modify: `docs/README.md`
- Modify: `tests/test_documentation.py`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: all current public contracts and the verified machine inspection.
- Produces: exact operational references without repeating handbook lessons.

- [ ] **Step 1: Add the failing reference contract test**

Create a mapping in `tests/test_documentation.py` that requires:

```python
REFERENCE_CONTRACTS = {
    "docs/reference/architecture.md": (
        "System context",
        "Package map",
        "Data flow",
        "Dependency rules",
        "Extension points",
    ),
    "docs/reference/configuration.md": (
        "View configuration",
        "Binary training configuration",
        "Sync training configuration",
        "CLI training arguments",
        "Validation rules",
        "Planned configuration files",
    ),
    "docs/reference/artifact-contracts.md": (
        "Manifest",
        "Split artifacts",
        "Cached clip",
        "Checkpoint",
        "Feature store",
        "Fusion artifact",
        "Predictions",
        "Threshold and metrics",
        "Hash relationships",
    ),
    "docs/reference/testing.md": (
        "Test layers",
        "Risk coverage",
        "Commands",
        "Fixtures",
        "Documentation checks",
        "Full-dataset checks",
    ),
    "docs/reference/hardware-and-compute.md": (
        "Verified machine",
        "Reliable hardware detection",
        "Storage policy",
        "Training policy",
        "Measurement policy",
        "Recheck commands",
    ),
}


@pytest.mark.parametrize(("relative", "headings"), REFERENCE_CONTRACTS.items())
def test_reference_contracts(relative: str, headings: tuple[str, ...]) -> None:
    assert_markdown_headings(Path.cwd(), relative, headings)
```

- [ ] **Step 2: Run the contract and confirm five missing files**

```powershell
uv run pytest tests\test_documentation.py -k reference_contract -v
```

Expected: five failures.

- [ ] **Step 3: Write architecture and configuration references**

`architecture.md` must include one package dependency diagram and one full data
flow. List every top-level package and its public responsibility. Document that
`cli.py` composes services but domain packages must not depend on the CLI.

`configuration.md` must tabulate every current field in `ViewConfig`,
`BinaryTrainingConfig`, `SyncTrainingConfig`, and branch CLI arguments. Include
type, default, unit, validation, and research effect. Keep planned YAML files in
a separate planned section.

- [ ] **Step 4: Write artifact and testing references**

`artifact-contracts.md` must tabulate every field in `ClipRecord`,
`PreparedClip`, `RunMetadata`, `FeatureRecord`, `FusionArtifact`, and evaluation
CSV rows. Show the hash chain and partition roles.

`testing.md` must map each test file to the research failure it prevents. List
focused, full, media, lint, formatting, lock, and documentation commands.

- [ ] **Step 5: Write the hardware reference**

Record:

- Ryzen 5 5600X, 6 cores, 12 threads.
- RTX 5070 Ti, 16,303 MiB from `nvidia-smi`.
- 32 GB RAM.
- Windows 11 Pro.
- Inspected free space by drive with the inspection date.

Explain that Windows reported an incorrect four GB adapter value. Include
PowerShell and `nvidia-smi` commands for rechecking. Do not invent throughput or
batch-size measurements.

- [ ] **Step 6: Link, validate, and commit**

```powershell
uv run pytest tests\test_documentation.py -v
uv run ddf-docs
git add CHANGELOG.md docs/README.md docs/reference tests/test_documentation.py
git commit -m "Add the technical reference set"
```

### Task 9: Research evidence layer

**Files:**
- Create: `docs/research/questions-and-hypotheses.md`
- Create: `docs/research/experiment-matrix.md`
- Create: `docs/research/metrics-and-statistics.md`
- Create: `docs/research/result-traceability.md`
- Create: `docs/research/findings.md`
- Create: `docs/research/error-analysis.md`
- Create: `docs/research/paper-outline.md`
- Modify: `docs/README.md`
- Modify: `tests/test_documentation.py`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: approved RQ1 through RQ5, selection rules, evaluation protocol, reproducibility contract, and threat model.
- Produces: preregistered comparisons and empty evidence ledgers that cannot be mistaken for completed results.

- [ ] **Step 1: Add failing research document contracts**

Require these exact headings:

```python
RESEARCH_CONTRACTS = {
    "docs/research/questions-and-hypotheses.md": (
        "Scope",
        "RQ1: Fusion generalization",
        "RQ2: Branch contribution",
        "RQ3: View integrity",
        "RQ4: Reliability",
        "RQ5: Cost",
        "Decision freeze",
    ),
    "docs/research/experiment-matrix.md": (
        "Experiment stages",
        "View comparisons",
        "Branch comparisons",
        "Fusion comparisons",
        "Stress and subgroup evaluations",
        "Compute limits",
        "Run status rules",
    ),
    "docs/research/metrics-and-statistics.md": (
        "Primary outcomes",
        "Metric definitions",
        "Bootstrap procedure",
        "Paired comparisons",
        "Multiple comparisons",
        "Reporting rules",
    ),
    "docs/research/result-traceability.md": (
        "Traceability contract",
        "Result registry",
        "Paper item rules",
    ),
    "docs/research/findings.md": (
        "Current evidence status",
        "Accepted findings",
        "Negative findings",
        "Superseded interpretations",
    ),
    "docs/research/error-analysis.md": (
        "Taxonomy",
        "Sampling protocol",
        "Reviewed cases",
        "Conclusions",
    ),
    "docs/research/paper-outline.md": (
        "Abstract evidence",
        "Introduction",
        "Related work",
        "Method",
        "Experiments",
        "Results",
        "Discussion",
        "Limitations",
        "Reproducibility appendix",
    ),
}


@pytest.mark.parametrize(("relative", "headings"), RESEARCH_CONTRACTS.items())
def test_research_contracts(relative: str, headings: tuple[str, ...]) -> None:
    assert_markdown_headings(Path.cwd(), relative, headings)
```

- [ ] **Step 2: Run the contracts and confirm seven missing files**

```powershell
uv run pytest tests\test_documentation.py -k research_contract -v
```

Expected: seven failures.

- [ ] **Step 3: Write questions, hypotheses, and the experiment matrix**

For each RQ, define a null hypothesis, alternative hypothesis, independent
variable, dependent variables, controls, primary validation metric, and allowed
conclusion.

The matrix must include identifiers for:

- `VIEW-DET-01`: MTCNN versus YuNet.
- `VIEW-TRACK-01`: greedy IoU versus motion-aware tracking.
- `VIEW-ALIGN-01`: box mouth crop versus landmark alignment.
- `VIS-01`: EfficientNet-B0 plus GRU versus ConvNeXt-Tiny.
- `AUD-01`: Wav2Vec2 Base versus WavLM.
- `AUD-02`: selected speech encoder versus AASIST.
- `SYNC-01`: current sync branch versus SyncNet-style baseline.
- `FUS-01`: single branches, pairs, and all branches.
- `FUS-02`: logistic regression versus MLP.
- `CAL-01`: Platt scaling versus eligible isotonic scaling.
- `REL-01`: abstention versus silent fallback.
- `SYNC-ABL-01`: authentic correspondence versus global fake labels.

Every row starts with status `planned`. Store no fake run IDs or results.

- [ ] **Step 4: Write statistics and traceability contracts**

Define source-grouped ROC-AUC as the primary fusion outcome. Define branch
primary metrics before their runs. Specify three seeds, 1,000 identity
bootstraps, paired fusion comparison, subgroup sample counts, and worst-method
reporting.

`result-traceability.md` must provide an empty registry with columns for result
ID, paper location, analysis command, report hash, prediction hash, MLflow run
IDs, checkpoint hashes, split hash, preprocessing hash, and Git commit.

- [ ] **Step 5: Write honest findings, error analysis, and paper mapping**

Begin `findings.md` with this exact sentence:

```text
No completed research findings are available.
```

Explain the evidence gate required to replace it. Leave accepted, negative, and
superseded sections empty except for instructions that do not look like results.

Define an error taxonomy for face absence, wrong identity, pose, occlusion,
compression, audio absence, clipping, duration mismatch, silence, sync crop,
generator family, calibration, and subgroup coverage. Keep the reviewed-cases
registry empty until sampling occurs.

Map each planned paper section to its source documents and future result IDs.

- [ ] **Step 6: Link, validate, and commit**

```powershell
uv run pytest tests\test_documentation.py -v
uv run ddf-docs
git add CHANGELOG.md docs/README.md docs/research tests/test_documentation.py
git commit -m "Define the research evidence system"
```

### Task 10: Architecture decision records

**Files:**
- Create: `docs/decisions/ADR-001-local-mlflow.md`
- Create: `docs/decisions/ADR-002-source-disjoint-splits.md`
- Create: `docs/decisions/ADR-003-calibrated-late-fusion.md`
- Create: `docs/decisions/ADR-004-quality-aware-abstention.md`
- Create: `docs/decisions/ADR-005-detector-bakeoff.md`
- Modify: `docs/README.md`
- Modify: `tests/test_documentation.py`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: current accepted design decisions and their documented alternatives.
- Produces: five immutable ADRs with explicit review triggers.

- [ ] **Step 1: Add the failing ADR contract test**

Require every ADR to contain:

```python
ADR_HEADINGS = (
    "Context",
    "Decision",
    "Options considered",
    "Consequences",
    "Review triggers",
)


@pytest.mark.parametrize(
    "name",
    (
        "ADR-001-local-mlflow.md",
        "ADR-002-source-disjoint-splits.md",
        "ADR-003-calibrated-late-fusion.md",
        "ADR-004-quality-aware-abstention.md",
        "ADR-005-detector-bakeoff.md",
    ),
)
def test_adr_contract(name: str) -> None:
    assert_markdown_headings(Path.cwd(), f"docs/decisions/{name}", ADR_HEADINGS)
```

- [ ] **Step 2: Run the ADR contract and confirm five missing files**

```powershell
uv run pytest tests\test_documentation.py::test_adr_contract -v
```

Expected: five failures.

- [ ] **Step 3: Write ADR-001 and ADR-002**

ADR-001 must compare local MLflow, hosted W&B, W&B offline, and no tracker.
Accept local MLflow because local ownership is a current requirement. Record
hosted supervisor sharing as the trigger to reconsider W&B before integration.

ADR-002 must compare row-random, source-disjoint, and fully identity-strict
splits. Accept source-disjoint as primary and identity-strict as a stress
subset. Explain the FakeAVCeleb source-target graph limitation.

- [ ] **Step 4: Write ADR-003 through ADR-005**

ADR-003 accepts calibrated logistic late fusion as primary. Record early fusion
and MLP alternatives, out-of-fold requirements, and the negative-result rule.

ADR-004 accepts explicit abstention when required evidence is missing. Explain
why full-frame fallback and probability imputation bias coverage.

ADR-005 does not select a detector. It accepts a controlled MTCNN versus YuNet
bakeoff, five-point landmark support, a reviewed training-only sample, and the
selection rules from `docs/model-selection.md`.

- [ ] **Step 5: Link, validate, and commit**

```powershell
uv run pytest tests\test_documentation.py -v
uv run ddf-docs
git add CHANGELOG.md docs/README.md docs/decisions tests/test_documentation.py
git commit -m "Record the initial architecture decisions"
```

### Task 11: CI integration and final documentation gate

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `.github/workflows/external-links.yml`
- Modify: `README.md`
- Modify: `ROADMAP.md`
- Modify: `docs/README.md`
- Modify: `docs/handbook/README.md`
- Modify: `docs/reference/testing.md`
- Modify: `tests/test_documentation.py`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: `ddf-docs`, the complete documentation tree, Ruff, pytest, and uv.
- Produces: one local verification command set and one CI workflow that enforce the same checks.

- [ ] **Step 1: Add failing integration assertions**

Add to `tests/test_documentation.py`:

```python
def test_root_readme_links_the_documentation_hub() -> None:
    text = Path("README.md").read_text(encoding="utf-8")
    assert "[Documentation handbook](docs/README.md)" in text


def test_ci_runs_every_required_quality_gate() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    for command in (
        "uv run ruff check src tests",
        "uv run ruff format --check src tests",
        "uv lock --check",
        "uv run ddf-docs",
        "uv run pytest",
    ):
        assert command in workflow


def test_external_links_run_outside_the_normal_ci_workflow() -> None:
    workflow = Path(".github/workflows/external-links.yml").read_text(
        encoding="utf-8"
    )
    assert "schedule:" in workflow
    assert "uv run ddf-docs --external" in workflow


def test_documentation_hub_covers_every_source_package() -> None:
    hub = Path("docs/README.md").read_text(encoding="utf-8")
    packages = {
        path.name
        for path in Path("src/deepfake_detection").iterdir()
        if path.is_dir() and not path.name.startswith("__")
    }
    assert all(f"`{package}`" in hub for package in packages)
```

- [ ] **Step 2: Run the integration assertions and confirm failures**

```powershell
uv run pytest tests\test_documentation.py -v
```

Expected: failure because the workflow and root handbook link do not exist.

- [ ] **Step 3: Add the CI workflow**

Create `.github/workflows/ci.yml` with a Windows job because the documented
commands use PowerShell and the main development target is Windows:

```yaml
name: ci

on:
  push:
  pull_request:

permissions:
  contents: read

jobs:
  test:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v6
        with:
          fetch-depth: 0
      - uses: astral-sh/setup-uv@c771a70e6277c0a99b617c7a806ffedaca235ff9
        with:
          enable-cache: true
          version: "0.12.1"
      - uses: actions/setup-python@v6
        with:
          python-version: "3.11"
      - name: Install
        run: uv sync --extra cpu --extra media --extra ml --group dev
      - name: Lint
        run: uv run ruff check src tests
      - name: Format
        run: uv run ruff format --check src tests
      - name: Lock
        run: uv lock --check
      - name: Documentation
        run: uv run ddf-docs
      - name: Pull request change contract
        if: github.event_name == 'pull_request'
        run: uv run ddf-docs --changed-from ${{ github.event.pull_request.base.sha }}
      - name: Tests
        run: uv run pytest
```

Install the Python media dependencies in the default CI job. The media test
still skips when the FFmpeg executable is unavailable.

Create `.github/workflows/external-links.yml` as a scheduled and manually
triggered workflow. Keep it separate from push and pull-request quality gates:

```yaml
name: external-links

on:
  schedule:
    - cron: "0 6 * * 1"
  workflow_dispatch:

permissions:
  contents: read

jobs:
  links:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v6
      - uses: astral-sh/setup-uv@c771a70e6277c0a99b617c7a806ffedaca235ff9
        with:
          enable-cache: true
          version: "0.12.1"
      - uses: actions/setup-python@v6
        with:
          python-version: "3.11"
      - name: Install
        run: uv sync --extra cpu --extra media --extra ml --group dev
      - name: Check external links
        run: uv run ddf-docs --external
```

- [ ] **Step 4: Complete root navigation and ownership**

Add `[Documentation handbook](docs/README.md)` near the start of `README.md`.
Keep the existing direct policy links.

Check `docs/README.md` against the actual top-level package directories. Add the
documentation package. Link all handbook, reference, research, ADR, and policy
documents. Remove all future path entries now that every target exists.

Update `docs/reference/testing.md` with the CI behavior and local equivalent.

- [ ] **Step 5: Update project controls**

Mark the three documentation tasks in Phase 1 of `ROADMAP.md` complete only
after every check below passes. Add one changelog entry covering the completed
handbook, references, research layer, ADRs, and CI gate.

- [ ] **Step 6: Run the complete verification set**

Run:

```powershell
uv run ruff check src tests
uv run ruff format --check src tests
uv lock --check
uv run ddf-docs
uv run pytest
git diff --check
git status --short
```

Expected:

- Ruff reports no violations.
- Ruff reports all Python files formatted.
- The lock resolves without changes.
- `ddf-docs` exits zero without output.
- Pytest reports all tests passing.
- Git reports only the intended Task 11 files before commit.

- [ ] **Step 7: Commit the documentation foundation**

```powershell
git add .github/workflows README.md ROADMAP.md CHANGELOG.md docs tests/test_documentation.py
git commit -m "Complete the documentation foundation"
git status --short --branch
```

Expected: the branch is `main` and the working tree is clean.

## Phase gate review

After Task 11, review these outcomes before planning MLflow:

- The documentation index has no dead links.
- A beginner can follow all 15 chapters in order.
- Every package has a teaching owner and reference owner.
- Current limitations are visible in the relevant chapters.
- Findings clearly state that no completed evidence exists.
- Experiment rows are planned and contain no invented run data.
- All five ADRs match the accepted project decisions.
- CI and local checks enforce the same project rules.

If any outcome fails, fix it within this phase. Do not begin the tracking and
configuration plan until the documentation foundation passes its gate.
