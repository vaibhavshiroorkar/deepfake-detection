from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .checks import (
    check_change_contract,
    check_cli_reference,
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
    issues.extend(check_cli_reference(root))
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
