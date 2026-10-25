"""LAV-DF head-training subset (D-058, E-011): videos of the official `train` split used to fit in-domain heads.

    python scripts/i6/build_lavdf_headtrain.py

N_PER_CELL videos for each of the four cells, drawn (seed 20261026) from the official train split after removing every video whose group
(source video) occurs in the fusion-fit subset `train` of lavdf_subset (D-057), so that the heads and the fusion layer never see the same source.
Output: data/_manifests/lavdf_headtrain.manifest.csv and data/_splits/lavdf_headtrain-<hash>.csv (every row `train`).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dfd.data import manifest as MF
from dfd.data import splits as SP

ROOT = Path("data")
META = ROOT / "lavdf" / "LAV-DF" / "metadata.min.json"
SEED = 20261026
N_PER_CELL = 2500
CELLS = {(False, False): "real", (True, False): "video_fake", (False, True): "audio_fake", (True, True): "both_fake"}


def main():
    meta = json.loads(META.read_text())
    sub = MF.read_manifest(ROOT / "_manifests" / "lavdf_subset.manifest.csv")
    split = SP.read_split(sorted((ROOT / "_splits").glob("lavdf_subset-*.csv"))[0])
    used_groups = {r.group_id for r in sub if split[r.sample_id] in ("train", "val")}
    used_files = {r.sample_id for r in sub}
    rng = np.random.default_rng(SEED)
    records, assignment = [], {}
    for (mv, ma), cell in CELLS.items():
        pool = sorted((r for r in meta if r["split"] == "train" and bool(r["modify_video"]) == mv and bool(r["modify_audio"]) == ma
                       and f"lavdf:{r['file']}" not in used_files and f"lavdf:{r['original'] or r['file']}" not in used_groups), key=lambda r: r["file"])
        for i in sorted(rng.choice(len(pool), size=N_PER_CELL, replace=False)):
            r = pool[i]
            sid = f"lavdf:{r['file']}"
            records.append(MF.Record(sample_id=sid, dataset="lavdf", family="lavdf", path=f"lavdf/LAV-DF/{r['file']}", modality="video", label=int(mv or ma),
                                     method=cell, group_id=f"lavdf:{r['original'] or r['file']}", group_basis="source_video", identity_known=False,
                                     official_split="train", notes=f"modify_video={int(mv)};modify_audio={int(ma)};n_fakes={r['n_fakes']};duration={r['duration']}"))
            assignment[sid] = "train"
    out = ROOT / "_manifests" / "lavdf_headtrain.manifest.csv"
    MF.write_manifest(out, records)
    assert not MF.validate_records(MF.read_manifest(out, validate=False))
    assert not [r.path for r in records if not (ROOT / r.path).is_file()]
    sid, p = SP.write_split(records, assignment, ROOT / "_splits", "lavdf_headtrain", dict(seed=SEED, n_per_cell=N_PER_CELL, excludes="groups of lavdf_subset train and val"), [out])
    print("manifest", out, len(records), "split", sid, "groups", len({r.group_id for r in records}))


if __name__ == "__main__":
    main()
