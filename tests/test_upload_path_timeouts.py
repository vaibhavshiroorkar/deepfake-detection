"""Every ffmpeg or ffprobe call an upload can reach has a time limit, so a hostile file cannot hang the server.

The website is reachable from the internet (docs/SECURITY.md). A call without a timeout on a crafted file can
run forever and hold the only scoring worker.
"""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "deepfake_detection"
UPLOAD_PATH = [
    "webapp/site.py",
    "webapp/provenance.py",
    "webapp/scorer.py",
    "preprocessing/neutralize.py",
    "avsync/mouth.py",
    "avsync/features.py",
    "views/media.py",
]


def _subprocess_calls(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("run", "Popen", "check_output", "call")
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "subprocess"
        ):
            yield node


@pytest.mark.parametrize("module", UPLOAD_PATH)
def test_every_subprocess_call_on_the_upload_path_has_a_timeout(module):
    path = SRC / module
    missing = [
        node.lineno
        for node in _subprocess_calls(path)
        if "timeout" not in {k.arg for k in node.keywords}
    ]
    assert not missing, f"{module}: subprocess calls without timeout at lines {missing}"
