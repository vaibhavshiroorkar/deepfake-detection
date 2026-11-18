"""Score the website calibration clips with the v2 candidates, from the study's preprocessed cache.

    .venv/Scripts/python.exe scripts/score_v2_cache.py [--device cuda] [--checks face,voice]

The calibration clips are those scripts/build_web_calibration.py reads: FakeAVCeleb identity-strict
test, DFDC, LAV-DF test (window label) and HiFi-AVDF. Their cleaned clips were preprocessed once for
P-001 (runs/p001/cache*), and the website's first part reproduces those inputs (tests/test_web_scorer.py),
so a candidate scored on the cached face crops and audio sees what it would see on the site. A clip is
scored by a check only when the study's model for that check scored it (the same clips can run).
Writes results/v2-eval/cache-scores.csv: corpus, clip_id, label, and one column per candidate (its
check score, a logit), appending and resuming by clip.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "results" / "v2-eval" / "cache-scores.csv"
INDEX = {
    "FakeAVCeleb": ("cache-index.csv", "cache"),
    "DFDC": ("cache-dfdc-index.csv", "cache-targets"),
    "LAV-DF": ("cache-lavdf-test-index.csv", "cache-targets"),
    "HiFi-AVDF": ("cache-hifiavdf-index.csv", "cache-targets"),
}


def main() -> int:
    from build_web_calibration import (  # the calibration's own clip list
        CORPORA,
        rows_for,
    )
    from score_v2_eval import candidates

    from deepfake_detection.data.cache_index import read_cache_index
    from deepfake_detection.views.cache_store import CacheStore
    from deepfake_detection.webapp.calibration import check_score

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checks", default="face,voice")
    arguments = parser.parse_args()
    models = candidates(arguments.device, set(arguments.checks.split(",")))
    names = [n for check in ("face", "voice") for n in models.get(check, {})]
    done = set()
    if OUT.exists():
        with open(OUT, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and reader.fieldnames[3:] != names:
                raise SystemExit(
                    f"{OUT.name} has columns {reader.fieldnames[3:]}, this run {names}: move it first"
                )
            done = {(r["corpus"], r["clip_id"]) for r in reader}
    new_file = not OUT.exists()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    began, n = time.time(), 0
    with open(OUT, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["corpus", "clip_id", "label", *names])
        for corpus, (branch_set, sync_stem) in CORPORA.items():
            index_file, cache = INDEX[corpus]
            index = read_cache_index(ROOT / "runs" / "p001" / index_file)
            store = CacheStore(ROOT / "runs" / "p001" / cache)
            sources: dict = {}
            rows = rows_for(corpus, branch_set, sync_stem, sources)
            clips = list(_clip_ids(sync_stem))
            for clip_id, row in zip(clips, rows, strict=True):
                if (corpus, clip_id) in done or clip_id not in index:
                    continue
                view = store.load(index[clip_id])
                values = []
                for check in ("face", "voice"):
                    for model in models.get(check, {}).values():
                        if row["scores"][check] is None:
                            values.append("")
                            continue
                        batch = _batch(view, check)
                        values.append(f"{check_score([model(batch)]):.6f}")
                writer.writerow([corpus, clip_id, row["label"], *values])
                n += 1
                if n % 200 == 0:
                    f.flush()
                    print(
                        f"{corpus} {n} clips, {(time.time() - began) / n:.2f} s/clip",
                        flush=True,
                    )
    print(f"done: {n} clips -> {OUT.relative_to(ROOT)}", flush=True)
    return 0


def _clip_ids(sync_stem: str):
    """Clip ids in the order rows_for reads them (the first seed's sync prediction file)."""
    path = ROOT / "results" / (sync_stem.format(seed=17) + "-predictions.csv")
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            yield r["clip_id"]


class _Batch:
    def __init__(self, values: torch.Tensor, valid: int) -> None:
        self.values = values
        self.valid_samples = torch.tensor([valid])


def _batch(view, check: str) -> _Batch:
    if check == "face":
        return _Batch(torch.from_numpy(view.visual_view).unsqueeze(0), 0)
    return _Batch(
        torch.from_numpy(view.audio_view).unsqueeze(0), int(view.audio_valid_samples)
    )


if __name__ == "__main__":
    raise SystemExit(main())
