"""E-029 split of the CommunityForensics-Small subset I6-S (preregistered rule), and the generator-overlap table against Eval.

Usage: python scripts/modern_image/e029_split.py
Fakes: by generator; within each architecture with at least 3 generators, model_names sorted by SHA-256 of the name, every fifth (positions
0, 5, ...) to validation; architectures with fewer than 3 generators stay in training. Reals: COCO and FFHQ by SHA-256 of the file name
(int of the first 8 hex digits mod 5 == 0 to validation); VISION by device id (D-number prefix) with the same rule. NSFW rows dropped.
Cap: at most 1,000 fake rows per (model_name, real_source) and split, numpy default_rng(20261027); reals are not capped (amendment A1).
Writes data/_results/modern/e029/split.csv and overlap.json. Reads only the metadata index.
"""
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib  # noqa: E402
import _env  # noqa: E402

CAP = 1000
RELATED = {"LCM_lora_sdv15": ["stable-diffusion-v1-5", "sd-v1-5", "sd15", "v1-5"], "LCM_lora_sdxl": ["stable-diffusion-xl", "sdxl"],
           "kvikontent_midjourney_v6": ["stable-diffusion-xl", "sdxl"], "LCM_lora_ssd1b": ["ssd-1b"], "kandinsky_2_2": ["kandinsky"],
           "stable_cascade": ["cascade", "wuerstchen"], "DeciDiffusionV2": ["decidiffusion"], "FLUX-dev": ["flux"], "FLUX-schnell": ["flux"],
           "Hourglass": ["hourglass"], "DFGAN": ["dfgan", "df-gan"], "GALIP": ["galip"], "Dalle2": ["dall-e", "dalle-2"],
           "Dalle3": ["dall-e", "dalle-3"], "MidjourneyV5_2": ["midjourney"], "MidjourneyV6_1": ["midjourney"]}


def h5(s):
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) % 5 == 0


def real_kind(name):
    p = re.sub(r"\d", "9", name)
    if p == "999999999999.jpg":
        return "coco"
    if p == "99999.png":
        return "ffhq"
    if p.startswith("D99_I"):
        return "vision"
    if p == "9999999.png":
        return "lhq"
    return "other"


def main():
    rows = lib.subset_rows("small")
    rows = [r for r in rows if r["nsfw_flag"] != "True"]
    fakes = [r for r in rows if r["label"] == "1"]
    gens_by_arch = defaultdict(set)
    for r in fakes:
        gens_by_arch[r["architecture"]].add(r["model_name"])
    val_gens = set()
    for a, gs in gens_by_arch.items():
        if len(gs) >= 3:
            order = sorted(gs, key=lambda g: hashlib.sha256(g.encode()).hexdigest())
            val_gens |= set(order[0::5])
    for r in rows:
        if r["label"] == "1":
            r["src"] = r["real_source"]
            r["split"] = "val" if r["model_name"] in val_gens else "train"
        else:
            k = real_kind(r["image_name"])
            r["src"] = k
            unit = r["image_name"].split("_")[0] if k == "vision" else r["image_name"]
            r["unit"] = unit
            r["split"] = "val" if h5(unit) else "train"
    rng = np.random.default_rng(lib.SEED)
    cell = defaultdict(list)
    for i, r in enumerate(rows):
        cell[(r["model_name"], r["src"], r["split"])].append(i)
    for k in sorted(cell):
        if k[0] in ("COCO", "FFHQ", "VISION"):
            continue  # amendment A1: the cap applies to fake rows only
        idx = cell[k]
        if len(idx) > CAP:
            keep = set(np.array(idx)[rng.permutation(len(idx))[:CAP]].tolist())
            for i in idx:
                if i not in keep:
                    rows[i]["split"] = "excluded_cap"
    out = _env.RES / "e029"
    out.mkdir(parents=True, exist_ok=True)
    cols = ["shard", "row", "image_name", "label", "model_name", "src", "architecture", "format", "resolution", "split"]
    with open(out / "split.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    c = Counter((r["split"], r["label"], r["architecture"] if r["label"] == "1" else r["src"]) for r in rows)
    train_gens = sorted({r["model_name"] for r in rows if r["label"] == "1" and r["split"] == "train"})
    eval_gens = sorted({r["model_name"] for r in lib.subset_rows("eval") if r["label"] == "1"})
    exact = [g for g in eval_gens if g in set(train_gens)]
    related = {g: sorted({t for t in train_gens for kw in RELATED.get(g, []) if kw in t.lower()}) for g in eval_gens}
    ov = {"n_train_generators": len(train_gens), "n_val_generators": len(val_gens), "val_generators": sorted(val_gens),
          "eval_generators": eval_gens, "exact_overlap": exact,
          "related_proxy": {g: v for g, v in related.items() if v}, "unrelated": [g for g in eval_gens if not related[g] and g not in exact],
          "keywords": RELATED, "counts": {"|".join(k): v for k, v in sorted(c.items())},
          "split_csv_sha256": hashlib.sha256(open(out / "split.csv", "rb").read()).hexdigest()}
    json.dump(ov, open(out / "overlap.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in ov.items() if k not in ("val_generators", "eval_generators")}, indent=1)[:3000])


if __name__ == "__main__":
    main()
