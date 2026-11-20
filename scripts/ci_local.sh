#!/usr/bin/env bash
# The CI checks of .github/workflows/ci.yml, run locally. The pre-push hook (.githooks/pre-push) runs it,
# so nothing reaches GitHub that CI would reject.
#
#     bash scripts/ci_local.sh
#
# It calls the project's .venv directly instead of `uv run`: `uv run` re-syncs the environment to the
# default extras and can replace the CUDA build of torch that the GPU jobs use.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PY=.venv/Scripts/python.exe
[ -x "$PY" ] || PY=.venv/bin/python


step() { printf '\n== %s\n' "$1"; }
started=$(date +%s)

step "Lint";          "$PY" -m ruff check src tests scripts
step "Format";        "$PY" -m ruff format --check src tests scripts
step "Lock";          uv lock --check
step "Documentation"; "$PY" -m deepfake_detection.documentation
# -P keeps the working folder off the import path, as the `pytest` command on GitHub does.
step "Tests";         "$PY" -P -m pytest -q -p no:cacheprovider
step "Browser behavior"; node --test tests/test_webapp_behavior.js
step "Tracked smoke"; "$PY" -m deepfake_detection.cli run --root . --config configs/local.yaml --config configs/smoke.yaml

printf '\nAll CI checks passed in %s s.\n' "$(( $(date +%s) - started ))"
