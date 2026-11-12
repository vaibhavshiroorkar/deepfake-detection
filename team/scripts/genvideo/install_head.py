"""Install the E-020 chosen head (C2) as the module's experimental head, with corrected identity and measured-evidence metadata.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/genvideo/install_head.py

Reads checkpoints/genvideo-E020/head_C2.npz (frozen, never modified; its SHA-256 is checked against the hash recorded before DF26 was read) and writes
checkpoints/genvideo-E020/head.npz with the same weights and a new meta block. Metadata corrections relative to head_C2.npz: protocol canonical-v1 was a
label error in the training script (the features came from data/_canonical_v3); the model identity, evaluation and scope text are added. Numbers are
copied from data/_results/genvideo/e020_dev_results.json and e020_df26_results.json, not typed. No weight, threshold or selection is changed.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CK = ROOT / "checkpoints" / "genvideo-E020"
RES = ROOT / "data" / "_results" / "genvideo"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    dev = json.loads((RES / "e020_dev_results.json").read_text())
    df = json.loads((RES / "e020_df26_results.json").read_text())
    src = CK / "head_C2.npz"
    assert sha(src) == df["head_hashes_at_dev"]["C2"] == dev["head_files"]["C2"], "head_C2 differs from the frozen hash"
    c2 = dev["candidates"]["C2"]
    d_all, d_cl = df["heads"]["C2"]["all"], df["heads"]["C2"]["closed"]
    z = np.load(src, allow_pickle=False)
    meta = json.loads(bytes(z["meta_json"]).decode())
    meta.update({
        "protocol": "canonical-v3",
        "head_id": "E020-C2-seed0",
        "validated": False,
        "source_head_sha256": sha(src),
        "model_name": "Appearance baseline: logistic regression on the mean DINOv2-small CLS embedding of 8 frames (NOT the ReStraV trajectory method)",
        "licence": "DINOv2-small Apache-2.0 (model card, DATA_LOG IMG-07); head trained by this project on DeepAction (generated folders CC BY 4.0, "
                   "real folder Pexels licence; DATA_LOG G1); evaluated on DF26 (evaluation only, DATA_LOG G2)",
        "score_meaning": "Uncalibrated ranking score in [0, 1] from a logistic regression on the mean CLS embedding of frozen DINOv2-small: higher means "
                         "the clip looks more like the DeepAction generated clips than like the DeepAction real stock clips. It is not a probability, and "
                         "on DF26 it ranked generated and real clips no better than chance.",
        "evaluation": f"E-020, experimental, NOT validated: DeepAction leave-one-generator-out macro AUC {c2['macro_logo_auc']} {c2['ci95']} (development); "
                      f"DF26 held-out AUC {d_all['auc']} {d_all['ci95']} on all {d_all['n']} clips and {d_cl['auc']} {d_cl['ci95']} on the closed-generator "
                      f"subset (preregistered H-G1 not supported)",
        "tested_scope": f"Tested only as E-020. On DeepAction (6 generators, 100 real stock clips from one library) the head reached a "
                        f"leave-one-generator-out macro AUC of {c2['macro_logo_auc']}. On the independent DF26 talking-head benchmark ({d_all['n']} clips, 271 real) its AUC "
                        f"was {d_all['auc']} (95% interval {d_all['ci95'][0]} to {d_all['ci95'][1]}), which is chance level; at the development threshold "
                        f"sensitivity was {df['heads']['C2']['sensitivity']} and specificity {df['heads']['C2']['specificity']}. It has not been shown to "
                        f"detect any generator it was not trained on.",
        "limitations": ["EXPERIMENTAL: the head did not generalise from DeepAction to DF26 (AUC about 0.5), so its output must not be used to decide whether a video is generated.",
                        "The DeepAction result is probably confounded by source and content (all real clips are stock footage from one library); this is a hypothesis, not tested.",
                        "It is an appearance classifier, not a temporal-artefact detector; the trajectory classifier (C1) was at chance on both corpora.",
                        "The score is not calibrated and the threshold was set on DeepAction only."],
    })
    arrs = {k: z[k] for k in z.files if k != "meta_json"}
    arrs["meta_json"] = np.frombuffer(json.dumps(meta, sort_keys=True).encode(), dtype=np.uint8)
    np.savez(CK / "head.npz", **arrs)
    print("wrote", CK / "head.npz", sha(CK / "head.npz"))


if __name__ == "__main__":
    main()
