"""Build the website's calibration file from committed P-001 results.

    .venv/Scripts/python.exe scripts/build_web_calibration.py

Calibration clips are the clips of the four P-001 evaluation sets: FakeAVCeleb identity-strict
test, DFDC, LAV-DF test under the window label (amendment A9) and HiFi-AVDF. Only their real
clips set the percentiles and the threshold; the fakes are counted to say how many the threshold
catches. Each check's score is the mean logit of its three seeds. The clip label comes from the
sync prediction files, which list every clip of a set, scored or not.

These clips are test data in the paper. The site uses them only to place a score, and the file
says so. Writes results/web-calibration.json.

With --choice results/web-models.json, the face and voice scores of the models it puts live replace
the study's, read from results/v2-eval/cache-scores.csv (scripts/score_v2_cache.py, the same cleaned
clips), and the file goes where the choice names it. The lip-sync scores stay the study's.
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
from deepfake_detection.webapp.calibration import (  # noqa: E402
    build,
    check_score,
    raised_by,
)

RESULTS = ROOT / "results"
OUTPUT = RESULTS / "web-calibration.json"
SEEDS = (17, 29, 43)
# corpus: (branch set name, sync prediction stem with {seed})
CORPORA = {
    "FakeAVCeleb": (
        "test-identity-strict",
        "p001-sync/sync-test-identity-strict-seed{seed}",
    ),
    "DFDC": ("dfdc", "p001-sync-targets/sync-dfdc-seed{seed}"),
    "LAV-DF": ("lavdf-test", "p001-sync-targets/sync-lavdf-window-seed{seed}"),
    "HiFi-AVDF": ("hifiavdf", "p001-sync-targets/sync-hifiavdf-seed{seed}"),
}
BRANCH_OF = {"face": "visual", "voice": "audio"}


def _read(path: Path, sources: dict) -> dict[str, dict]:
    sources[str(path.relative_to(ROOT).as_posix())] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
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
    sync = [
        _read(RESULTS / (sync_stem.format(seed=s) + "-predictions.csv"), sources)
        for s in SEEDS
    ]
    branches = {
        check: [
            _read(
                RESULTS / "p001-branches" / f"{b}-seed{s}-{branch_set}-predictions.csv",
                sources,
            )
            for s in SEEDS
        ]
        for check, b in BRANCH_OF.items()
    }
    rows = []
    for clip, row in sync[0].items():
        scores = {check: _score(tables, clip) for check, tables in branches.items()}
        scores["sync"] = _score(sync, clip)
        rows.append(
            {
                "corpus": corpus,
                "clip_id": clip,
                "label": int(row["label"]),
                "scores": scores,
            }
        )
    return rows


def replace_scores(rows: list[dict], choice: dict, sources: dict) -> None:
    """Put the chosen models' cache scores in place of the study's face and voice scores, and add the
    generated-video check's. Each check's entry may name its score file ("scores"); the default is
    results/v2-eval/cache-scores.csv."""
    tables: dict[str, dict] = {}
    for check in ("face", "voice", "gen"):
        if check not in choice:
            continue
        name = choice[check].get("scores", "results/v2-eval/cache-scores.csv")
        if name not in tables:
            tables[name] = {
                (r["corpus"], r["clip_id"]): r for r in _read_rows(ROOT / name, sources)
            }
        for row in rows:
            cached = tables[name].get((row["corpus"], row["clip_id"]))
            value = None if cached is None else cached.get(choice[check]["column"], "")
            row["scores"][check] = float(value) if value else None


def add_extra_reals(rows: list[dict], path: Path, sources: dict) -> None:
    """Real-world real clips (scripts/build_realworld_reals.py), each source its own corpus, real only.

    The research corpora hold studio, broadcast and actor footage; a visitor's webcam or phone recording
    read as fake against them alone (found 11-19). These rows only set the lines; they hold no fakes.
    """

    def value(text: str) -> float | None:
        return float(text) if text not in ("", None) else None

    for r in _read_rows(path, sources):
        rows.append(
            {
                "corpus": r["corpus"],
                "clip_id": r["clip_id"],
                "label": 0,
                "scores": {
                    "face": value(r.get("face")),
                    "voice": value(r.get("voice")),
                    "sync": value(r.get("sync")),
                    "gen": value(r.get("gen")),
                },
            }
        )


def _read_rows(path: Path, sources: dict) -> list[dict]:
    name = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
    sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--choice", type=Path, default=None)
    arguments = parser.parse_args()
    sources: dict[str, str] = {}
    rows = [
        r for corpus, (b, s) in CORPORA.items() for r in rows_for(corpus, b, s, sources)
    ]
    output = OUTPUT
    choice = None
    if arguments.choice is not None:
        choice = json.loads(arguments.choice.read_text(encoding="utf-8"))
        replace_scores(rows, choice, sources)
        if choice.get("extra_reals"):
            add_extra_reals(rows, ROOT / choice["extra_reals"], sources)
        output = ROOT / choice["calibration"]["video"]
    calibration, summary = build(rows)
    coverage = {
        c: {
            check: sum(
                r["scores"].get(check) is not None for r in rows if r["corpus"] == c
            )
            for check in ("face", "voice", "sync", "gen")
        }
        | {"clips": sum(r["corpus"] == c for r in rows)}
        for c in dict.fromkeys(r["corpus"] for r in rows)
    }
    meta = {
        "built": date.today().isoformat(),
        "code_commit": _git_commit(),
        "note": "Calibration clips are P-001 test data; the site uses them only to place scores.",
        "check_score": "mean logit of the three seeds",
        "flag_quantile": 0.95,
        "unsure_quantile": 0.80,
        "coverage": coverage,
        "outcomes": summary,
        "raised_by": raised_by(calibration, rows),
        "sources": sources,
    }
    if choice is not None:
        meta["models"] = {c: choice[c] for c in ("face", "voice", "gen") if c in choice}
    data = json.loads(calibration.to_json())
    data["meta"] = meta
    output.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    print(
        f"threshold {calibration.threshold:.4f}, can't tell above {calibration.lower:.4f} -> {output.relative_to(ROOT)}"
    )
    print(json.dumps({"coverage": coverage, "outcomes": summary}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
