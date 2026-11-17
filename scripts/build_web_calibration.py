"""Build the website's calibration file from committed P-001 results.

    .venv/Scripts/python.exe scripts/build_web_calibration.py

Calibration clips are the clips of the four P-001 evaluation sets: FakeAVCeleb identity-strict
test, DFDC, LAV-DF test under the window label (amendment A9) and HiFi-AVDF. Only their real
clips set the percentiles and the threshold; the fakes are counted to say how many the threshold
catches. Each check's score is the mean logit of its three seeds. The clip label comes from the
sync prediction files, which list every clip of a set, scored or not.

These clips are test data in the paper. The site uses them only to place a score, and the file
says so. Writes results/web-calibration.json.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deepfake_detection.cli.common import _git_commit  # noqa: E402
from deepfake_detection.webapp.calibration import build, check_score  # noqa: E402

RESULTS = ROOT / "results"
OUTPUT = RESULTS / "web-calibration.json"
SEEDS = (17, 29, 43)
# corpus: (branch set name, sync prediction stem with {seed})
CORPORA = {
    "FakeAVCeleb": ("test-identity-strict", "p001-sync/sync-test-identity-strict-seed{seed}"),
    "DFDC": ("dfdc", "p001-sync-targets/sync-dfdc-seed{seed}"),
    "LAV-DF": ("lavdf-test", "p001-sync-targets/sync-lavdf-window-seed{seed}"),
    "HiFi-AVDF": ("hifiavdf", "p001-sync-targets/sync-hifiavdf-seed{seed}"),
}
BRANCH_OF = {"face": "visual", "voice": "audio"}


def _read(path: Path, sources: dict) -> dict[str, dict]:
    sources[str(path.relative_to(ROOT).as_posix())] = hashlib.sha256(path.read_bytes()).hexdigest()
    with open(path, encoding="utf-8", newline="") as f:
        return {r["clip_id"]: r for r in csv.DictReader(f)}


def _probability(row: dict | None) -> float | None:
    if row is None or row.get("probability", "") in ("", "nan", "NaN"):
        return None
    value = float(row["probability"])
    return None if math.isnan(value) else value


def _score(tables: list[dict], clip: str) -> float | None:
    """The check score if every seed scored the clip, else None (the check could not run)."""
    probs = [_probability(t.get(clip)) for t in tables]
    return None if any(p is None for p in probs) else check_score(probs)


def rows_for(corpus: str, branch_set: str, sync_stem: str, sources: dict) -> list[dict]:
    sync = [_read(RESULTS / (sync_stem.format(seed=s) + "-predictions.csv"), sources) for s in SEEDS]
    branches = {check: [_read(RESULTS / "p001-branches" / f"{b}-seed{s}-{branch_set}-predictions.csv", sources)
                        for s in SEEDS] for check, b in BRANCH_OF.items()}
    rows = []
    for clip, row in sync[0].items():
        scores = {check: _score(tables, clip) for check, tables in branches.items()}
        scores["sync"] = _score(sync, clip)
        rows.append({"corpus": corpus, "label": int(row["label"]), "scores": scores})
    return rows


def main() -> int:
    sources: dict[str, str] = {}
    rows = [r for corpus, (b, s) in CORPORA.items() for r in rows_for(corpus, b, s, sources)]
    calibration, summary = build(rows)
    coverage = {c: {check: sum(r["scores"][check] is not None for r in rows if r["corpus"] == c)
                    for check in ("face", "voice", "sync")} | {"clips": sum(r["corpus"] == c for r in rows)}
                for c in CORPORA}
    meta = {
        "built": date.today().isoformat(),
        "code_commit": _git_commit(),
        "note": "Calibration clips are P-001 test data; the site uses them only to place scores.",
        "check_score": "mean logit of the three seeds",
        "flag_quantile": 0.95, "unsure_quantile": 0.80,
        "coverage": coverage, "outcomes": summary, "sources": sources,
    }
    data = json.loads(calibration.to_json())
    data["meta"] = meta
    OUTPUT.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    print(f"threshold {calibration.threshold:.4f}, can't tell above {calibration.lower:.4f} -> {OUTPUT.relative_to(ROOT)}")
    print(json.dumps({"coverage": coverage, "outcomes": summary}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
