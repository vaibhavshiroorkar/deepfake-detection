"""Evaluation groups for re-analysing the stored E-014 sealed-test scores in the Evaluation Lab (D-096).

Three sealed cross-dataset blocks of the frozen iteration I-4 get five predeclared groups each, like the LAV-DF groups of E-033:
In-the-Wild (speech head T-A), Celeb-DF v2 (face-video head T-V) and the ForenSynths CNN-generator subset (image head T-I).

Reads ONLY the columns sample_id, group_id and label of the seed-0 rows of data/_final/scores/I-4.csv (the logged E-014 score
file, ledger sha256 e0c78a8cd1b1...). The score column is never read, so no group can be chosen by how the model did on it.
Writes new files only: data/_manifests/sealed_reanalysis/e014_sealed.groups.json and one descriptor per dataset under
data/_evallab/imports/sealed/. Refuses to overwrite a frozen groups file with different content.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/posteval/build_sealed_groups.py
"""
from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_manifests import group_entry, sha256_file, write_once  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SCORES = ROOT / "data" / "_final" / "scores" / "I-4.csv"
OUT = ROOT / "data" / "_manifests" / "sealed_reanalysis"
DESC_DIR = ROOT / "data" / "_evallab" / "imports" / "sealed"
SEED = 20261031
SPLIT = "test_x"

THRESHOLD = {"source": "val_fit", "value": None,
             "provenance": "per-seed threshold of maximum balanced accuracy on the validation rows of the same runs (D-024), fitted at the "
                           "freeze before the sealed split was read; the rule E-014 used; not tuned on this data"}
SCORE_KIND = ("score of the frozen I-4 seed runs exactly as logged at the freeze in data/_final/scores/I-4.csv; on a dataset the model "
              "was not trained on it is a ranking score, not a calibrated probability")
COMMON_CAVEATS = [
    "Re-analysis of stored scores: the sealed split was read once, at the freeze (E-014). Nothing is scored again and no model runs. "
    "A result here is a view of the logged evaluation, not a new held-out test, and must not feed back into any model or threshold choice.",
    "Groups were fixed from sample ids, group ids and labels only (seed 20261031) without reading any score, and are frozen by SHA-256.",
    "The interval is a group bootstrap over the independent groups of the selected rows; small groups give wide intervals.",
]

SPECS = {
    "in_the_wild": {
        "modality": "audio", "head": "T-A", "unit": "speaker",
        "title": "In-the-Wild speech deepfakes (sealed test, E-014): speech head (T-A)",
        "run_ids": ["I-4-audio-s0", "I-4-audio-s1", "I-4-audio-s2"],
        "eligibility": "cross-corpus: the speech head was trained and calibrated on ASVspoof 2019 LA only; In-the-Wild was never used "
                       "for training, tuning or selection. The sealed part is speaker-disjoint from the development slice.",
        "caveats": ["42 speakers: the bootstrap resamples speakers, so even the whole set has a wide interval.",
                    "The head scores about the first 4 s of each recording."],
        "sizes": [("G1", "Small", 100), ("G2", "Medium", 500), ("G3", "Large", 2000)],
    },
    "celebdf_v2": {
        "modality": "visual", "head": "T-V", "unit": "identity group",
        "title": "Celeb-DF v2 face swaps (sealed test, E-014): face-video head (T-V)",
        "run_ids": ["I-4-visual-s0", "I-4-visual-s1", "I-4-visual-s2"],
        "eligibility": "cross-dataset: the face-video head was trained and calibrated on FaceForensics++ only; Celeb-DF v2 was never "
                       "used for training, tuning or selection. The sealed part is identity-disjoint from the development slice.",
        "caveats": ["Celeb-DF licence: research use only; no clip or derived data from it may be published."],
        "sizes": [("G1", "Small", 50), ("G2", "Medium", 200), ("G3", "Large", 600)],
    },
    "forensynths_cnn_subset": {
        "modality": "image", "head": "T-I", "unit": "generator group",
        "title": "ForenSynths CNN-generated images, 6 unseen generators (sealed test, E-014): image head (T-I)",
        "run_ids": ["I-4-image-s0", "I-4-image-s1", "I-4-image-s2"],
        "eligibility": "cross-generator: the image head was trained and calibrated on ProGAN images only; none of these six "
                       "generators was used for training, tuning or selection.",
        "caveats": ["16 independent groups (a generator, or a generator and image category): the bootstrap resamples these, so intervals are wide."],
        "sizes": [("G1", "Small", 100), ("G2", "Medium", 500), ("G3", "Large", 2000)],
    },
}


def read_rows() -> dict:
    """dataset -> rows {sample_id, group_id, label} of the seed-0 run; the score column is not read."""
    want = {ds: spec["run_ids"][0] for ds, spec in SPECS.items()}
    out = defaultdict(list)
    with open(SCORES, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["split"] == SPLIT and want.get(r["dataset"]) == r["run_id"]:
                out[r["dataset"]].append({"sample_id": r["sample_id"], "group_id": r["group_id"], "label": int(r["label"])})
    for ds in SPECS:
        out[ds].sort(key=lambda r: r["sample_id"])
    return out


def class_ranks(rows: list) -> None:
    rng = random.Random(SEED)
    for lab in (0, 1):
        ids = [r for r in rows if r["label"] == lab]
        rng.shuffle(ids)
        for i, r in enumerate(ids):
            r["class_rank"] = i


def diagnostic(ds: str, rows: list) -> dict:
    if ds == "in_the_wild":
        by = defaultdict(lambda: ([], []))
        for r in sorted(rows, key=lambda r: r["class_rank"]):
            by[r["group_id"]][r["label"]].append(r)
        sub = []
        for real, fake in by.values():
            k = min(len(real), len(fake))
            sub += real[:k] + fake[:k]
        return group_entry("G5", "Diagnostic: balanced within each speaker", "diagnostic",
                           "Every speaker with both real and fake recordings, with the same number of each (first in the seeded order). "
                           "Inside this group a speaker's identity carries no information about the label, so it tests whether the head "
                           "separates real from fake speech of the same person rather than one speaker from another.",
                           sub, "label", "overlaps G1 to G4", "speaker")
    if ds == "celebdf_v2":
        sub = [r for r in rows if not r["sample_id"].split(":", 1)[1].startswith("YouTube-real/")]
        return group_entry("G5", "Diagnostic: same-source reals only (YouTube-real left out)", "diagnostic",
                           "Celeb-real against Celeb-synthesis only. The YouTube-real videos come from a different source than the fakes; "
                           "without them the real and fake videos come from the same pool of celebrity interviews, so a difference in "
                           "source cannot separate the classes.", sub, "label", "overlaps G1 to G4", "identity group")
    i2i = ("cyclegan", "gaugan", "deepfake")
    sub = [r for r in rows if r["group_id"].split(":")[1] in i2i]
    return group_entry("G5", "Diagnostic: image-to-image generators only", "diagnostic",
                       "CycleGAN, GauGAN and Deepfake with their own real images. These generators transform or condition on a real "
                       "input, so the fakes keep the layout of real photographs; it tests whether detection holds when the fake is "
                       "built from a real image rather than drawn from noise.", sub, "label", "overlaps G1 to G4", "generator group")


def main() -> int:
    data = read_rows()
    datasets, counts = [], {}
    for ds, spec in SPECS.items():
        rows = data[ds]
        if not rows:
            raise SystemExit(f"no rows for {ds} {SPLIT} in {SCORES}")
        class_ranks(rows)
        gs = []
        for gid, title, k in spec["sizes"]:
            sub = [r for r in rows if r["class_rank"] < k]
            gs.append(group_entry(gid, f"{title} sample", "general",
                                  f"The first {k} real and the first {k} fake items of the seeded order (seed {SEED}, separate order per class).",
                                  sub, "label", "nested: contains every smaller group", spec["unit"]))
        gs.append(group_entry("G4", "Whole sealed test set (the E-014 block)", "general",
                              "Every item of the sealed block, exactly the rows E-014 reported (classes not balanced; the share is shown).",
                              rows, "label", "contains every group", spec["unit"]))
        gs.append(diagnostic(ds, rows))
        datasets.append({"dataset": ds, "title": spec["title"], "experiment": "E-014",
                         "modality": spec["modality"], "head": spec["head"], "label_column": "label", "split": SPLIT,
                         "groups": gs, "full": None})
        counts[ds] = {"rows": len(rows), "by_label": dict(Counter(r["label"] for r in rows)), "groups": len({r["group_id"] for r in rows})}
    payload = {"experiment": "E-014", "purpose": "re-analysis of stored sealed scores (D-096)", "seed": SEED,
               "source": "data/_final/scores/I-4.csv: columns sample_id, group_id, label of the seed-0 runs; the score column is not read",
               "score_file_sha256": sha256_file(SCORES), "counts": counts, "datasets": datasets}
    gname = "e014_sealed.groups.json"
    gsha = write_once(OUT / gname, json.dumps(payload, indent=1, sort_keys=True) + "\n")
    DESC_DIR.mkdir(parents=True, exist_ok=True)
    for d in datasets:
        spec = SPECS[d["dataset"]]
        desc = {"schema": "evallab-groups/1", "dataset": d["dataset"], "title": spec["title"],
                "experiment": "E-014", "iteration": "I-4", "modality": spec["modality"], "head": spec["head"], "split": SPLIT,
                "sealed_reanalysis": True, "run_ids": spec["run_ids"], "score_file": "data/_final/scores/I-4.csv",
                "groups_file": f"data/_manifests/sealed_reanalysis/{gname}", "groups_sha256": gsha, "bootstrap_seed": 20261024,
                "threshold": THRESHOLD, "score_kind": SCORE_KIND, "eligibility": spec["eligibility"],
                "caveats": COMMON_CAVEATS + spec["caveats"],
                "provenance": {"experiment": "E-014", "report": "data/_final/eval/I-4.json", "ledger": "ledger/final_ledger.jsonl",
                               "score_file_sha256": payload["score_file_sha256"], "decision": "D-096"}}
        p = DESC_DIR / f"E-014_{spec['head'].replace('-', '')}.lab.json"
        text = json.dumps(desc, indent=1) + "\n"
        if p.exists() and p.read_text(encoding="utf-8") != text:
            raise SystemExit(f"refused: {p} exists with different content")
        p.write_text(text, encoding="utf-8", newline="\n")
        for g in d["groups"]:
            print(d["dataset"], g["id"], g["n_samples"], "real", g["n_real"], "fake", g["n_fake"], "groups", g["n_groups"])
    print("groups.json sha256", gsha)
    return 0


if __name__ == "__main__":
    sys.exit(main())
