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
