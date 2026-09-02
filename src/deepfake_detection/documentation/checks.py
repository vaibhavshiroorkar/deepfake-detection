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


IGNORED_PARTS = frozenset({".git", ".pytest_cache", ".venv", "data"})
INCOMPLETE_MARKERS = ("[IN" + "COMPLETE]", "PENDING" + " CONTENT")
FORBIDDEN_CHARACTERS = frozenset(
    {"\u2013", "\u2014", "\u2022", "\u00b7", "\u2026", "\u2192"}
)
LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
FENCED_CODE_PATTERN = re.compile(
    r"^(?P<backtick>`{3,})[^\n]*\n.*?^(?P=backtick)`*[ \t]*$"
    r"|^(?P<tilde>~{3,})[^\n]*\n.*?^(?P=tilde)~*[ \t]*$",
    re.DOTALL | re.MULTILINE,
)
INLINE_CODE_PATTERN = re.compile(r"`[^`\n]+`")
CLI_BLOCK_START = "<!-- BEGIN GENERATED COMMANDS -->"
CLI_BLOCK_END = "<!-- END GENERATED COMMANDS -->"


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


def git_changed_paths(root: Path, base_ref: str) -> tuple[Path, ...]:
    result = subprocess.run(  # noqa: S603
        ["git", "-C", str(root), "diff", "--name-only", f"{base_ref}...HEAD"],  # noqa: S607
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
    request = Request(  # noqa: S310
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
    issues: list[DocumentationIssue] = []
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
