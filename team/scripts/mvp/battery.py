"""I-0 diagnostic battery (EXPERIMENTS.md section "Diagnostic battery"), CPU only, train, validation and dev_x data only.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/mvp/battery.py

Reads data/_results/I-0/<task>/scores.npz and metrics.json (written by train_mvp.py), the manifests, the face statistics of face_stats.py and
the checkpoint head. Writes data/_results/I-0/battery.md (generated, not edited by hand), battery.json, figures in data/_results/I-0/figs/
and contact sheets of the worst errors in data/_results/I-0/worst/ (local only, never committed: dataset licences). All AUC values carry the
95 percent group-bootstrap interval of the protocol (2,000 resamples) over the 3 seeds; the shipped head (seed 0, calibrated) is used for the
histograms, the calibration curve and the worst errors. Items 1 to 8 of the battery are named in the section titles.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from dfd.eval import bootstrap as B  # noqa: E402
from dfd.eval import calibration as C  # noqa: E402
from dfd.eval import metrics as M  # noqa: E402
from dfd.infer.checkpoint import load_checkpoint  # noqa: E402
from dfd.sweep.rank import load_split  # noqa: E402
from dfd.sweep.tasks import TASKS  # noqa: E402

OUT = Path("data/_results/I-0")
FIGS = OUT / "figs"
CKPT = "checkpoints/I-0-mvp"
WIN = {"T-V": "IMG-07", "T-I": "IMG-13", "T-A": "AU-8"}
MANIFESTS = {"T-V": ["ffpp", "celebdf_v2"], "T-I": ["forensynths_progan_val", "forensynths_cnn_subset"], "T-A": ["asvspoof2019_la", "in_the_wild"]}
L: list[str] = []


def md(s=""):
    L.append(s)


def load_manifest(names):
    out = {}
    for n in names:
        for r in csv.DictReader(open(f"data/_manifests/{n}.manifest.csv", encoding="utf-8")):
            out[r["sample_id"]] = r
    return out


def auc_ci(y, S, g, n_boot=B.N_BOOT):
    """S: (n_seeds, n). Returns dict or None when a class is missing."""
    if len(np.unique(y)) < 2 or len(np.unique(g)) < 2:
        return None
    b = B.cluster_bootstrap(y, S, g, "auc", n_boot=n_boot)
    return dict(auc=float(b.estimate), low=float(b.low), high=float(b.high), seed_sd=b.seed_sd)


def image_ci(y, s, n_boot=2000, seed=20261024):
    rng = np.random.default_rng(seed)
    i0, i1 = np.nonzero(y == 0)[0], np.nonzero(y == 1)[0]
    reps = [M.roc_auc(y[idx], s[idx]) for idx in (np.concatenate([rng.choice(i0, i0.size), rng.choice(i1, i1.size)]) for _ in range(n_boot))]
    return dict(auc=float(M.roc_auc(y, s)), low=float(np.quantile(reps, 0.025)), high=float(np.quantile(reps, 0.975)))


def fmt(c, n=3):
    return "n/a" if c is None else f"{c['auc']:.{n}f} [{c['low']:.{n}f}, {c['high']:.{n}f}]"


def subset_auc(y, S, g, mask, label="subset"):
    if mask.sum() < 10:
        return None
    return auc_ci(y[mask], S[:, mask], g[mask])


def per_method(split_ids, y, S, g, man, key, title):
    """AUC of (real vs each method) with the group bootstrap; real examples are shared by all rows."""
    meth = np.array([man[s][key] for s in split_ids])
    rows = []
    for m in sorted(set(meth) - {"real"}):
        k = (meth == m) | (meth == "real")
        rows.append((m, int((meth == m).sum()), fmt(subset_auc(y, S, g, k))))
    md(f"| {title} | fakes | AUC [95% interval] |")
    md("|---|---|---|")
    for r in rows:
        md(f"| {r[0]} | {r[1]} | {r[2]} |")
    md()
    return {r[0]: r[2] for r in rows}


def cal_bins(y, p, n_bins=15):
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        k = idx == b
        rows.append((edges[b], edges[b + 1], int(k.sum()), float(p[k].mean()) if k.any() else None, float(y[k].mean()) if k.any() else None))
    return rows


def contact_sheet(paths, scores, labels, out, cols=5, size=180):
    import cv2
    tiles = []
    for p, s, l in zip(paths, scores, labels):
        img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((size, size, 3), np.uint8)
        h, w = img.shape[:2]
        sc = size / max(h, w)
        img = cv2.resize(img, (max(1, int(w * sc)), max(1, int(h * sc))), interpolation=cv2.INTER_AREA)
        canvas = np.full((size + 18, size, 3), 255, np.uint8)
        canvas[:img.shape[0], :img.shape[1]] = img
        cv2.putText(canvas, f"p={s:.2f} y={l}", (3, size + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        tiles.append(canvas)
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".jpg", np.vstack(rows))[1].tofile(str(out))


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    (OUT / "worst").mkdir(exist_ok=True)
    mt = json.loads((OUT / "metrics.json").read_text())
    meta, heads = load_checkpoint(CKPT)
    res = {}
    md("# I-0 diagnostic battery (generated by scripts/mvp/battery.py; do not edit by hand)")
    md()
    md(f"Checkpoint {mt['checkpoint_id']}. Winners: T-V IMG-07, T-I IMG-13, T-A AU-8. Seeds 0, 1, 2; 95 percent group-bootstrap intervals (2,000 resamples). "
       "Validation = the in-dataset validation split of the task, dev_x = the development slice (exploratory, D-041). Nothing from test_in or test_x is read.")
    md()
    for tid in ("T-V", "T-I", "T-A"):
        z = np.load(OUT / tid / "scores.npz", allow_pickle=False)
        m = mt["tasks"][tid]
        man = load_manifest(MANIFESTS[tid])
        T0 = m["per_seed"][0]["temperature"]
        md(f"## {tid}: {WIN[tid]}")
        md()
        res[tid] = {}
        # ------------------------------------------------ items 2, 3, 8
        md("### Items 2, 3 and 8: gaps and seed variance")
        md()
        md("| split | n clips | AUC [95% interval] | seed sd | interval half-width |")
        md("|---|---|---|---|---|")
        for k in ("train", "val", "dev"):
            a = m[f"{k}_auc"]
            md(f"| {k} | {m['n'][k]} | {a['estimate']:.4f} [{a['low']:.4f}, {a['high']:.4f}] | {a['seed_sd']:.4f} | {(a['high'] - a['low']) / 2:.4f} |")
        md()
        md(f"Train minus validation AUC: {m['gap_train_minus_val']:+.4f}. Validation minus dev_x AUC: {m['gap_val_minus_dev']:+.4f}. "
           f"Seed sd on dev_x {m['seed_sd_dev_auc']:.4f} against a bootstrap half-width of {m['dev_auc_half_width']:.4f}.")
        md()
        # ------------------------------------------------ item 1
        md("### Item 1: per-dataset and per-method or per-generator AUC")
        md()
        vk, dk = ok_idx(z, "val"), ok_idx(z, "dev")
        if tid == "T-V":
            md("Validation (FF++ c23, real against each manipulation method):")
            md()
            per_method(z["val_sample_id"][vk], z["val_y"][vk], z["val_scores"][:, vk], z["val_group"][vk], man, "method", "method")
            md("dev_x (Celeb-DF v2, 15 identity groups): " + fmt(auc_ci(z["dev_y"][dk], z["dev_scores"][:, dk], z["dev_group"][dk])) + ". Celeb-DF has one fake method label (celebdf_synthesis).")
            md()
        elif tid == "T-I":
            cls = np.array([z["val_sample_id"][i].split(":", 1)[1].split("/")[0] for i in range(len(z["val_y"]))])
            md("Validation (ProGAN, 6 held-out LSUN classes, real against fake per class; image-level interval inside each class, one group each):")
            md()
            md("| class | n | AUC [95% interval] |")
            md("|---|---|---|")
            for c in sorted(set(cls)):
                k = cls == c
                md(f"| {c} | {int(k.sum())} | {fmt(image_ci(z['val_y'][k], z['val_scores'].mean(0)[k]))} |")
            md()
            md("dev_x (two CNN-synthesis generator units, image-level interval inside each unit, which is one group each):")
            md()
            md("| unit | n | AUC [95% interval] |")
            md("|---|---|---|")
            for gname in np.unique(z["dev_group"]):
                k = z["dev_group"] == gname
                md(f"| {str(gname).split(':', 1)[1]} | {int(k.sum())} | {fmt(image_ci(z['dev_y'][k], z['dev_scores'].mean(0)[k]))} |")
            md()
        else:
            md("Validation (ASVspoof 2019 LA dev, real against each attack A01 to A06):")
            md()
            per_method(z["val_sample_id"][vk], z["val_y"][vk], z["val_scores"][:, vk], z["val_group"][vk], man, "method", "attack")
            md("dev_x (In-the-Wild, 12 speakers): " + fmt(auc_ci(z["dev_y"][dk], z["dev_scores"][:, dk], z["dev_group"][dk])) + ". Per speaker (real and fake of the same speaker; image-level interval within the speaker):")
            md()
            md("| speaker | n | real | fake | AUC [95% interval] |")
            md("|---|---|---|---|---|")
            sp = {}
            for gname in np.unique(z["dev_group"]):
                k = z["dev_group"] == gname
                y_, s_ = z["dev_y"][k], z["dev_scores"].mean(0)[k]
                sp[str(gname)] = image_ci(y_, s_, 500) if len(np.unique(y_)) == 2 else None
                md(f"| {str(gname).split(':', 1)[1]} | {int(k.sum())} | {int((y_ == 0).sum())} | {int((y_ == 1).sum())} | {fmt(sp[str(gname)])} |")
            md()
            vals = [v["auc"] for v in sp.values() if v]
            md(f"Per-speaker AUC: minimum {min(vals):.3f}, median {np.median(vals):.3f}, maximum {max(vals):.3f}; speakers with AUC below 0.5: {sum(v < 0.5 for v in vals)} of {len(vals)}.")
            md()
        # ------------------------------------------------ item 4
        md("### Item 4: calibration (15 equal-width bins), ECE and Brier before and after temperature scaling")
        md()
        md("| split | ECE before / after | Brier before / after | mean temperature |")
        md("|---|---|---|---|")
        for k in ("val", "dev"):
            md(f"| {k} | {m['mean'][k]['ece_before']:.4f} / {m['mean'][k]['ece_after']:.4f} | {m['mean'][k]['brier_before']:.4f} / {m['mean'][k]['brier_after']:.4f} | {m['temperature_mean']:.3f} |")
        md()
        fig, axes = plt.subplots(2, 2, figsize=(8, 6.4))
        for r_, k in enumerate(("val", "dev")):
            ok = ok_idx(z, k)
            y, p = z[f"{k}_y"][ok], z[f"{k}_scores"][0][ok]
            pc = C.sigmoid(C.logit(p) / T0)
            for c_, (pp, name) in enumerate(((p, "before temperature scaling"), (pc, "after"))):
                ax = axes[r_, c_]
                bins = cal_bins(y, pp)
                xs = [b[3] for b in bins if b[3] is not None]
                ys = [b[4] for b in bins if b[3] is not None]
                ax.plot([0, 1], [0, 1], color="#888", lw=0.8)
                ax.plot(xs, ys, "o-", ms=3)
                ax.set_title(f"{tid} {k}, {name}", fontsize=8)
                ax.set_xlabel("mean predicted probability of fake", fontsize=7)
                ax.set_ylabel("fraction fake", fontsize=7)
        fig.tight_layout()
        fig.savefig(FIGS / f"calibration_{tid}.png", dpi=120)
        plt.close(fig)
        res[tid]["calibration"] = {k: cal_bins(z[f"{k}_y"][ok_idx(z, k)], z[f"{k}_scores"][0][ok_idx(z, k)]) for k in ("val", "dev")}
        # ------------------------------------------------ item 5
        fig, axes = plt.subplots(1, 3, figsize=(11, 3))
        for ax, k in zip(axes, ("train", "val", "dev")):
            ok = ok_idx(z, k)
            y, p = z[f"{k}_y"][ok], C.sigmoid(C.logit(z[f"{k}_scores"][0][ok]) / T0)
            ax.hist(p[y == 0], bins=30, range=(0, 1), alpha=0.6, label="real", density=True)
            ax.hist(p[y == 1], bins=30, range=(0, 1), alpha=0.6, label="fake", density=True)
            ax.set_title(f"{tid} {k} (calibrated, seed 0)", fontsize=8)
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIGS / f"hist_{tid}.png", dpi=120)
        plt.close(fig)
        md(f"### Item 5: score histograms: figs/hist_{tid}.png (real and fake, train, validation, dev_x); calibration curves: figs/calibration_{tid}.png")
        md()
        yd, pd_ = z["dev_y"][dk], C.sigmoid(C.logit(z["dev_scores"][0][dk]) / T0)
        yv, pv_ = z["val_y"][vk], C.sigmoid(C.logit(z["val_scores"][0][vk]) / T0)
        md("Quantiles of the calibrated score (shipped head), 10 / 50 / 90 percent:")
        md()
        md("| split | real | fake |")
        md("|---|---|---|")
        for name, y, p in (("validation", yv, pv_), ("dev_x", yd, pd_)):
            q = lambda a: " / ".join(f"{v:.3f}" for v in np.quantile(a, [0.1, 0.5, 0.9]))  # noqa: E731
            md(f"| {name} | {q(p[y == 0])} | {q(p[y == 1])} |")
        md()
        # ------------------------------------------------ item 6 (T-V only: face statistics; others: what exists)
        md("### Item 6: splits by compression, face size and number of faces")
        md()
        if tid == "T-V":
            fs_file = OUT / "T-V" / "face_stats.json"
            vz, _ = load_split(WIN[tid], TASKS[tid], TASKS[tid].val)
            dz, _ = load_split(WIN[tid], TASKS[tid], TASKS[tid].devx)
            for name, sp_, y, S, g, sid in (("validation", vz, z["val_y"], z["val_scores"], z["val_group"], z["val_sample_id"]),
                                            ("dev_x", dz, z["dev_y"], z["dev_scores"], z["dev_group"], z["dev_sample_id"])):
                nm = np.load(f"data/_features/{WIN[tid]}/T-V/{'ffpp_val' if name == 'validation' else 'celebdf_dev_x'}.npz", allow_pickle=False)["mask"]
                nf = 16 - nm.sum(1)
                md(f"{name}: frames without a face per clip: mean {nf.mean():.2f}, clips with at least one such frame {int((nf > 0).sum())} of {len(nf)}, clips with no face at all {int((nf == 16).sum())}.")
            md()
            md("Compression: FF++ c23 only (one level); the other FF++ levels are not part of the training data of this iteration, and the compression of Celeb-DF v2 is not recorded in its manifest, so no split by compression can be made (stated, not skipped silently).")
            md()
            if fs_file.is_file():
                fs = json.loads(fs_file.read_text())
                for name, y, S, g, sid in (("validation", z["val_y"], z["val_scores"], z["val_group"], z["val_sample_id"]), ("dev_x", z["dev_y"], z["dev_scores"], z["dev_group"], z["dev_sample_id"])):
                    ok = ok_idx(z, "val" if name == "validation" else "dev")
                    rec = [fs.get(s, {}) for s in sid]
                    nfaces = np.array([r.get("n_faces", -1) for r in rec])
                    fh = np.array([r["face_h_rel"] if r.get("face_h_rel") is not None else np.nan for r in rec])
                    md(f"By number of faces in three sampled frames, {name}:")
                    md()
                    md("| faces | clips | real | fake | AUC [95% interval] |")
                    md("|---|---|---|---|---|")
                    for lab, k in (("0", nfaces == 0), ("1", nfaces == 1), ("2 or more", nfaces >= 2)):
                        k = k & ok
                        md(f"| {lab} | {int(k.sum())} | {int((y[k] == 0).sum())} | {int((y[k] == 1).sum())} | {fmt(subset_auc(y, S, g, k))} |")
                    md()
                    ok2 = ok & ~np.isnan(fh)
                    if ok2.sum() > 30:
                        q1, q2 = np.quantile(fh[ok2], [1 / 3, 2 / 3])
                        md(f"By face height relative to the frame height (largest face, tertiles {q1:.3f} and {q2:.3f}), {name}:")
                        md()
                        md("| tertile | clips | real | fake | AUC [95% interval] |")
                        md("|---|---|---|---|---|")
                        for lab, k in (("small", fh <= q1), ("middle", (fh > q1) & (fh <= q2)), ("large", fh > q2)):
                            k = k & ok2
                            md(f"| {lab} | {int(k.sum())} | {int((y[k] == 0).sum())} | {int((y[k] == 1).sum())} | {fmt(subset_auc(y, S, g, k))} |")
                        md()
            else:
                md("Face statistics file missing (scripts/mvp/face_stats.py not run).")
                md()
            # aggregation diagnostics (frame level against clip level)
            md("Frame-level against clip-level (aggregation), shipped head, dev_x and validation:")
            md()
            md("| split | frame-level AUC | clip mean (shipped) | clip median | clip top-3 mean | clip max | within-clip sd of frame scores, real / fake |")
            md("|---|---|---|---|---|---|---|")
            for name, sp_, y, g in (("validation", vz, vz.y, vz.groups), ("dev_x", dz, dz.y, dz.groups)):
                p = heads["T-V"].prob_rows(sp_.X)
                clip = sp_.clip_of
                n = y.size
                order = [np.nonzero(clip == i)[0] for i in range(n)]
                have = np.array([o.size > 0 for o in order])
                mean_ = np.array([p[o].mean() if o.size else np.nan for o in order])
                med_ = np.array([np.median(p[o]) if o.size else np.nan for o in order])
                top3 = np.array([np.sort(p[o])[-3:].mean() if o.size else np.nan for o in order])
                mx = np.array([p[o].max() if o.size else np.nan for o in order])
                sd_ = np.array([p[o].std() if o.size > 1 else np.nan for o in order])
                fr = auc_ci(y[clip], p[None], g[clip])
                cols = [fmt(auc_ci(y[have], v[None, have], g[have])) for v in (mean_, med_, top3, mx)]
                md(f"| {name} | {fmt(fr)} | " + " | ".join(cols) + f" | {np.nanmean(sd_[y == 0]):.3f} / {np.nanmean(sd_[y == 1]):.3f} |")
            md()
        elif tid == "T-I":
            md("T-I: the dev_x units differ in image size and content (see the per-unit rows in item 1); no compression level, face size or face count applies to the whole-image path. The no-face and multiple-face counts of D-034 belong to the video and face-image paths.")
            md()
        else:
            md("T-A: the splits by clip duration, leading silence, level and repeat-padded fraction are the trivial-feature measurement M-004 (data/_results/M-004/m004.json); compression levels are not recorded for ASVspoof 2019 or In-the-Wild.")
            md()
        # ------------------------------------------------ item 7
        md("### Item 7: the 20 worst errors on dev_x (10 false alarms, 10 misses), shipped head, calibrated scores")
        md()
        ok = dk
        y = z["dev_y"][ok]
        p = C.sigmoid(C.logit(z["dev_scores"][0][ok]) / T0)
        sid = z["dev_sample_id"][ok]
        grp = z["dev_group"][ok]
        fa = np.argsort(-np.where(y == 0, p, -1))[:10]
        ms = np.argsort(np.where(y == 1, p, 2))[:10]
        md("| kind | sample | group | score | method or note |")
        md("|---|---|---|---|---|")
        worst = []
        for kind, ids in (("false alarm", fa), ("miss", ms)):
            for i in ids:
                note = man[sid[i]].get("method", "")
                md(f"| {kind} | {sid[i]} | {grp[i]} | {p[i]:.3f} | {note} |")
                worst.append((sid[i], p[i], int(y[i])))
        md()
        groups_w = defaultdict(int)
        for i in list(fa) + list(ms):
            groups_w[str(grp[i])] += 1
        md("Worst errors per group: " + ", ".join(f"{k.split(':', 1)[1]} {v}" for k, v in sorted(groups_w.items(), key=lambda kv: -kv[1])) + ".")
        md()
        if tid == "T-V":
            paths = []
            for s, _, _ in worst:
                d = Path("data/_crops/celebdf_v2_yunet_n16") / __import__("dfd.prep.pipeline", fromlist=["folder_name"]).folder_name(s)
                paths.append(d / "8.jpg")
            contact_sheet(paths, [w[1] for w in worst], [w[2] for w in worst], OUT / "worst" / "T-V_dev_worst.jpg")
        elif tid == "T-I":
            paths = [Path("data") / man[s]["path"] for s, _, _ in worst]
            contact_sheet(paths, [w[1] for w in worst], [w[2] for w in worst], OUT / "worst" / "T-I_dev_worst.jpg")
        res[tid]["worst"] = worst
    (OUT / "battery.md").write_text("\n".join(L), encoding="utf-8")
    (OUT / "battery.json").write_text(json.dumps(res, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print("battery written", OUT / "battery.md")


def ok_idx(z, k):
    return ~np.isnan(z[f"{k}_scores"]).any(0)


if __name__ == "__main__":
    main()
