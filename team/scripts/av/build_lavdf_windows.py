"""LAV-DF windows for the audio-visual task T-AV (manifest and split), D-100.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/av/build_lavdf_windows.py

LAV-DF edits a short span (often under one second) inside a longer clip, and the
I-6 heads learned to tell produced videos from originals instead of finding the
edit. So every sampled fake file gives two 2 s windows:

    forged    centred on its first fake period, label 1
    genuine   centred in the longest stretch of the same file that keeps MARGIN
              seconds away from every fake period, label 0 (method 'real')

The pair shares speaker, microphone, codec and production, so only the edit
separates them. A fake file without a stretch of WINDOW seconds is skipped.
Original real files give one window each at a seeded start, label 0, so the
module also sees unproduced video. `notes` records the window start, the kind
(forged, genuine, original) and the cell, so the within-produced control of
I-6 can be computed on genuine against forged windows alone.

Official train -> train, dev -> val, test -> test_in (refused by the loaders
unless final=True). Group = the source video, so a fake, its genuine window
and its real source share a group. The test rows are fixed here and not read
until a T-AV iteration is frozen.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dfd.data import manifest as MF
from dfd.data import splits as SP

ROOT = Path("data")
META = ROOT / "lavdf" / "LAV-DF" / "metadata.min.json"
SEED = 20261102
WINDOW = 2.0
MARGIN = 0.2
OFFICIAL = {"train": "train", "dev": "val", "test": "test_in"}
CELLS = {(True, False): "video_fake", (False, True): "audio_fake", (True, True): "both_fake"}
# Fake files per cell and original real files per official split.
N_FAKE = {"train": 4000, "dev": 600, "test": 1000}
N_REAL = {"train": 6000, "dev": 900, "test": 1500}


def forged_start(row: dict) -> float:
    begin, end = row["fake_periods"][0]
    return float(np.clip((begin + end) / 2 - WINDOW / 2, 0.0, max(row["duration"] - WINDOW, 0.0)))


def genuine_start(row: dict) -> float | None:
    """Start of a WINDOW centred in the longest clean stretch, or None if none is long enough."""
    edges = [0.0]
    for begin, end in sorted(row["fake_periods"]):
        edges += [begin - MARGIN, end + MARGIN]
    edges.append(row["duration"])
    gaps = [(edges[i], edges[i + 1]) for i in range(0, len(edges), 2)]
    begin, end = max(gaps, key=lambda g: g[1] - g[0])
    if end - begin < WINDOW:
        return None
    return round((begin + end) / 2 - WINDOW / 2, 3)


def record(row: dict, start: float, label: int, kind: str, cell: str, split: str) -> MF.Record:
    source = row["original"] or row["file"]
    return MF.Record(
        sample_id=f"lavdf:{row['file']}@{start:.3f}", dataset="lavdf", family="lavdf", path=f"lavdf/LAV-DF/{row['file']}",
        modality="video", label=label, method=cell if label else "real", group_id=f"lavdf:{source}",
        group_basis="source_video", identity_known=False, official_split=split,
        notes=f"start={start:.3f};length={WINDOW};kind={kind};cell={cell};"
              f"modify_video={int(row['modify_video'])};modify_audio={int(row['modify_audio'])}",
    )


def main() -> None:
    meta = json.loads(META.read_text())
    rng = np.random.default_rng(SEED)
    records, assignment, skipped = [], {}, {}
    for official, ours in OFFICIAL.items():
        rows = sorted((r for r in meta if r["split"] == official), key=lambda r: r["file"])
        for (video, audio), cell in CELLS.items():
            pool = [r for r in rows if r["n_fakes"] > 0 and bool(r["modify_video"]) == video and bool(r["modify_audio"]) == audio]
            pool = [r for r in pool if r["duration"] >= WINDOW]
            pairable = [r for r in pool if genuine_start(r) is not None]
            skipped[f"{official}/{cell}"] = len(pool) - len(pairable)
            for i in sorted(rng.choice(len(pairable), size=min(N_FAKE[official], len(pairable)), replace=False)):
                r = pairable[i]
                records.append(record(r, forged_start(r), 1, "forged", cell, official))
                records.append(record(r, genuine_start(r), 0, "genuine", cell, official))
        real = [r for r in rows if r["n_fakes"] == 0 and r["duration"] >= WINDOW]
        for i in sorted(rng.choice(len(real), size=min(N_REAL[official], len(real)), replace=False)):
            r = real[i]
            start = round(float(rng.uniform(0.0, r["duration"] - WINDOW)), 3)
            records.append(record(r, start, 0, "original", "real", official))
        for r in records:
            if r.official_split == official:
                assignment[r.sample_id] = ours
    out = ROOT / "_manifests" / "av_lavdf_windows.manifest.csv"
    MF.write_manifest(out, records)
    errors = MF.validate_records(MF.read_manifest(out, validate=False))
    assert not errors, errors[:3]
    missing = [r.path for r in records if not (ROOT / r.path).is_file()]
    assert not missing, missing[:3]
    split_id, _ = SP.write_split(
        records, assignment, ROOT / "_splits", "av_lavdf_windows",
        dict(seed=SEED, window_s=WINDOW, margin_s=MARGIN, n_fake_per_cell=N_FAKE, n_real=N_REAL,
             fake_files_without_clean_stretch=skipped, source="metadata.min.json official split", map=OFFICIAL),
        [out],
    )
    print("manifest", out, len(records), "rows; split", split_id)
    print("fake files skipped for lack of a clean stretch:", skipped)


if __name__ == "__main__":
    main()
