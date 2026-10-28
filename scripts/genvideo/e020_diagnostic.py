"""Descriptive post-hoc diagnostic of the frozen E-020 C2 head (no fitting, no selection, no threshold change; output is not used for any decision).

    PYTHONPATH=src .venv/Scripts/python.exe scripts/genvideo/e020_diagnostic.py

Shows (1) how the frozen head's scores are distributed on DeepAction and DF26 (saturation), (2) how far the DF26 embeddings are from the DeepAction
training embeddings in the head's own standardised space, (3) whether the DF26 score depends on a simple content cue (mean luminance) rather than the label.
"""
import json
from pathlib import Path

import numpy as np

from dfd.platform.genvideo.model import load_head

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "data" / "_results" / "genvideo"


def q(x):
    return [round(float(v), 4) for v in np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])]


def rank_corr(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return round(float(np.corrcoef(ra, rb)[0, 1]), 4)


def main():
    h = load_head(ROOT / "checkpoints" / "genvideo-E020" / "head_C2.npz")
    out = {"head": "head_C2.npz (frozen)"}
    zs = {}
    for name in ("deepaction", "df26"):
        z = np.load(RES / f"features_{name}.npz", allow_pickle=False)
        x, y = z["cls_mean"].astype(np.float64), z["label"]
        s = h.score(x)
        zs[name] = ((x - h.mean) / np.maximum(h.std, 1e-8), s, y, z["stats"])
        out[name] = {"n": int(len(y)), "score_quantiles_real": q(s[y == 0]), "score_quantiles_fake": q(s[y == 1]),
                     "fraction_saturated_real": round(float(((s < 1e-3) | (s > 1 - 1e-3))[y == 0].mean()), 4),
                     "fraction_saturated_fake": round(float(((s < 1e-3) | (s > 1 - 1e-3))[y == 1].mean()), 4),
                     "mean_abs_standardised_embedding": round(float(np.abs(zs[name][0]).mean()), 4)}
    zd, _, _, _ = zs["deepaction"]
    zf, sf, yf, stf = zs["df26"]
    out["df26_embedding_offset_in_deepaction_sd"] = round(float(np.linalg.norm(zf.mean(0) - zd.mean(0)) / np.sqrt(zd.shape[1])), 4)
    out["df26_score_vs_mean_luminance_spearman"] = rank_corr(sf, stf[:, 4])
    out["df26_label_vs_mean_luminance_spearman"] = rank_corr(yf.astype(float), stf[:, 4])
    (RES / "e020_diagnostic.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
