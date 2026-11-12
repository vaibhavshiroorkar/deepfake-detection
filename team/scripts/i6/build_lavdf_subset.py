"""LAV-DF fixed subset for the fusion iteration (manifest and split), D-057.

    python scripts/i6/build_lavdf_subset.py

Reads data/lavdf/LAV-DF/metadata.min.json (official split train / dev / test, two booleans per file: modify_video, modify_audio). Draws a fixed
stratified sample, N_PER_CELL videos for each of the four cells (real, video fake only, audio fake only, both) in each official split, with a
fixed seed. Official `dev` becomes `val`, official `test` becomes `test_in` (the loaders refuse it unless final=True). Group = the source video
(`original` of a fake, the file itself for a real one), so a fake and its real source share a group; the official split is identity-disjoint
by the authors' rule (identities are not in the metadata, DATA_LOG.md). Fused label: fake if modify_video or modify_audio. The per-modality
labels are written to the `notes` column and the cell name to `method`.
Output: data/_manifests/lavdf_subset.manifest.csv, data/_splits/lavdf_subset-<hash>.csv and .json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dfd.data import manifest as MF
from dfd.data import splits as SP

ROOT = Path("data")
META = ROOT / "lavdf" / "LAV-DF" / "metadata.min.json"
SEED = 20261025
N_PER_CELL = 500
OFFICIAL = {"train": "train", "dev": "val", "test": "test_in"}
CELLS = {(False, False): "real", (True, False): "video_fake", (False, True): "audio_fake", (True, True): "both_fake"}


def main():
    meta = json.loads(META.read_text())
    rng = np.random.default_rng(SEED)
    records, assignment = [], {}
    for off, ours in OFFICIAL.items():
        for (mv, ma), cell in CELLS.items():
            pool = sorted((r for r in meta if r["split"] == off and bool(r["modify_video"]) == mv and bool(r["modify_audio"]) == ma), key=lambda r: r["file"])
            pick = rng.choice(len(pool), size=N_PER_CELL, replace=False)
            for i in sorted(pick):
                r = pool[i]
                src = r["original"] or r["file"]
                sid = f"lavdf:{r['file']}"
                records.append(MF.Record(sample_id=sid, dataset="lavdf", family="lavdf", path=f"lavdf/LAV-DF/{r['file']}", modality="video",
                                         label=int(mv or ma), method=cell, group_id=f"lavdf:{src}", group_basis="source_video", identity_known=False,
                                         official_split=off, notes=f"modify_video={int(mv)};modify_audio={int(ma)};n_fakes={r['n_fakes']};duration={r['duration']}"))
                assignment[sid] = ours
    out = ROOT / "_manifests" / "lavdf_subset.manifest.csv"
    MF.write_manifest(out, records)
    errs = MF.validate_records(MF.read_manifest(out, validate=False))
    assert not errs, errs[:3]
    missing = [r.path for r in records if not (ROOT / r.path).is_file()]
    assert not missing, missing[:3]
    sid, p = SP.write_split(records, assignment, ROOT / "_splits", "lavdf_subset",
                            dict(seed=SEED, n_per_cell=N_PER_CELL, cells=list(CELLS.values()), source="metadata.min.json official split", map=OFFICIAL), [out])
    print("manifest", out, len(records), "split", sid)


if __name__ == "__main__":
    main()
