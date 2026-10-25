"""Score file of one frozen iteration (the staircase of PROTOCOL.md section 2 addendum, D-041).

    # smoke test, never reads a sealed split (val, dev_x and the LAV-DF fusion-fit split only), small:
    PYTHONPATH=src .venv/Scripts/python.exe scripts/freeze/score_final.py --iteration I-0 --dry-run --limit 200
    # the real run at the freeze (all splits including the sealed ones; refuses without --final):
    PYTHONPATH=src .venv/Scripts/python.exe scripts/freeze/score_final.py --iteration I-1 --final

Iterations (cumulative configurations, see docs/_agents/FREEZE.md):
  I-0   T-V IMG-07 frozen + linear heads, T-I IMG-13 + head, T-A AU-8 first-4-seconds window + head          (E-004)
  I-1   T-V IMG-17 fine-tuned; T-I, T-A as I-0                                                                (E-005)
  I-2   T-A AU-8 on the trimmed, peak-normalised window (head retrained); T-V, T-I as I-1                     (E-007)
  I-3   T-A head trained with the three augmented copies; T-V, T-I as I-1                                     (E-008)
  I-4   T-A logit mean of AU-8, AU-4, AU-6; T-V, T-I as I-1                                                   (E-009)
  I-5   LAV-DF subset, frozen heads of E-005 (visual) and E-009 (audio)                                       (E-010)
  I-6   LAV-DF subset, in-domain heads of E-011 and the logistic-regression fusion                            (E-011)
  I-6b  LAV-DF subset, visual head trained on produced videos only                                           (E-012)
  BASE  T-V Xception baseline of D-025/D-060                                                                  (E-013)
The score file has exactly the columns of dfd.eval.scorefile (run_id, iteration, seed, modality, dataset, split, sample_id, group_id, label, score),
raw probabilities, one row per clip or image, seeds 0, 1, 2. The evaluation (dfd.eval.run --final --iteration <tag>) is a separate step.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "i6"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as K  # noqa: E402
from common import SEEDS, Block, NpHead, logit, sigmoid  # noqa: E402

ITERATIONS = ["I-0", "I-1", "I-2", "I-3", "I-4", "I-5", "I-6", "I-6b", "BASE"]
COLUMNS = ["run_id", "iteration", "seed", "modality", "dataset", "split", "sample_id", "group_id", "label", "score"]
HEADS = Path("data/_final/heads")
ENSEMBLE = ["AU-8", "AU-4", "AU-6"]
HEAD_TAG = {"trim_norm_i2": "I-2", "trim_norm_i3": "I-3"}


class Run:
    def __init__(self, args):
        self.args = args
        self.final = bool(args.final)
        self.dry = not self.final
        self.limit = None if self.final else args.limit
        self.root = K.FINAL_ROOT if self.final else K.DRY_ROOT
        self.files: dict[str, str] = {}        # provenance: path -> sha256
        self.notes: list[str] = []

    def use(self, path):
        p = str(path)
        if p not in self.files:
            self.files[p] = K.sha256(p)
        return path

    def guarded(self, what, fn):
        """In a dry run a block that needs the GPU or a missing file is skipped with a note; in a final run any failure stops the run."""
        if self.final:
            return fn()
        try:
            return fn()
        except Exception as e:   # noqa: BLE001
            msg = f"skipped in the dry run ({what}): {type(e).__name__}: {str(e)[:150]}"
            print("  " + msg, flush=True)
            self.notes.append(msg)
            return []


# ------------------------------------------------------------------ T-V
def blocks_video_frozen(run: Run):
    out = []
    for ds in K.VIDEO_SETS:
        for split in K.splits_of(ds, run.dry):
            d = K.video_features("IMG-07", ds, split, run.final, run.limit, run.root)
            for s in SEEDS:
                h = NpHead(run.use(HEADS / f"I-0_T-V_s{s}.npz"))
                out.append(Block("visual", ds["key"], split, s, d["sample_id"], d["group_id"], d["label"], K.frozen_video_p(d["feats"], d["mask"], h), d["n_listed"]))
    return out


def blocks_video_ft(run: Run, cid: str, model_of_seed, weights_of_seed):
    out = []
    for s in SEEDS:
        jobs = [(ds, split, K.select(ds, split, run.final, "video", run.limit)) for ds in K.VIDEO_SETS for split in K.splits_of(ds, run.dry)]
        res = K.ft_scores(model_of_seed(s), cid, str(run.use(weights_of_seed(s))), jobs, run.root)
        for (key, split), d in res.items():
            out.append(Block("visual", key, split, s, d["sample_id"], d["group_id"], d["label"], d["p"], d["n_listed"]))
    return out


def derived_celebdf(blocks):
    """Celeb-DF++ has no real video of its own (D-045 item 3); the AUC on it is taken against the Celeb-DF v2 reals (item 4), identity as the group."""
    def g(arr):
        return np.array(["celebdf:" + str(x).rsplit(":", 1)[-1] for x in arr])
    out = []
    for s in SEEDS:
        v2 = [b for b in blocks if b.dataset == "celebdf_v2" and b.split == "test_x" and b.seed == s and b.modality == "visual"]
        pp = [b for b in blocks if b.dataset == "celebdf_pp" and b.split == "test_x" and b.seed == s and b.modality == "visual"]
        if not (v2 and pp):
            continue
        v2, pp = v2[0], pp[0]
        r = v2.label == 0
        out.append(Block("visual", "celebdf_pp_with_v2_real", "test_x", s, np.r_[pp.sample_id, v2.sample_id[r]], np.r_[g(pp.group_id), g(v2.group_id[r])],
                         np.r_[pp.label, v2.label[r]], np.r_[pp.p, v2.p[r]], pp.listed + int(r.sum())))
        out.append(Block("visual", "celebdf_family", "test_x", s, np.r_[v2.sample_id, pp.sample_id], np.r_[g(v2.group_id), g(pp.group_id)],
                         np.r_[v2.label, pp.label], np.r_[v2.p, pp.p], v2.listed + pp.listed))
    return out


# ------------------------------------------------------------------ T-I
def blocks_image(run: Run):
    out = []
    for ds in K.IMAGE_SETS:
        for split in K.splits_of(ds, run.dry):
            d = K.image_features("IMG-13", ds, split, run.final, run.limit, run.root)
            for s in SEEDS:
                h = NpHead(run.use(HEADS / f"I-0_T-I_s{s}.npz"))
                out.append(Block("image", ds["key"], split, s, d["sample_id"], d["group_id"], d["label"], h.prob_rows(d["feats"]), d["n_listed"]))
    return out


# ------------------------------------------------------------------ T-A
def blocks_audio(run: Run, kind: str):
    """kind: first (I-0, I-1), trim_norm_i2, trim_norm_i3, ensemble (I-4)."""
    out = []
    for ds in K.AUDIO_SETS:
        for split in K.splits_of(ds, run.dry):
            if kind == "first":
                feats = {"AU-8": K.audio_features("AU-8", "first", ds, split, run.final, run.limit, run.root)}
                heads = {s: [NpHead(run.use(HEADS / f"I-0_T-A_s{s}.npz"))] for s in SEEDS}
            elif kind in ("trim_norm_i2", "trim_norm_i3"):
                feats = {"AU-8": K.audio_features("AU-8", "trim_norm", ds, split, run.final, run.limit, run.root)}
                heads = {s: [NpHead(run.use(HEADS / f"{HEAD_TAG[kind]}_T-A_s{s}.npz"))] for s in SEEDS}
            else:
                feats = {c: K.audio_features(c, "trim_norm", ds, split, run.final, run.limit, run.root) for c in ENSEMBLE}
                heads = {s: [NpHead(run.use(Path("data/_results/I-4") / f"head_{c}_s{s}.npz")) for c in ENSEMBLE] for s in SEEDS}
            first = next(iter(feats.values()))
            assert all(np.array_equal(f["sample_id"], first["sample_id"]) for f in feats.values()), "members cover different items"
            for s in SEEDS:
                z = np.mean([h.logit_rows(f["feats"]) for h, f in zip(heads[s], feats.values())], axis=0)
                out.append(Block("audio", ds["key"], split, s, first["sample_id"], first["group_id"], first["label"], sigmoid(z), first["n_listed"]))
    return out


# ------------------------------------------------------------------ LAV-DF (I-5, I-6, I-6b)
def lavdf_meta(recs):
    notes = [dict(kv.split("=") for kv in r.notes.split(";")) for r in recs]
    return dict(sample_id=np.array([r.sample_id for r in recs]), group_id=np.array([r.group_id for r in recs]), y_fused=np.array([r.label for r in recs]),
                y_video=np.array([int(n["modify_video"]) for n in notes]), y_audio=np.array([int(n["modify_audio"]) for n in notes]),
                cell=np.array([r.method for r in recs]))


def lavdf_audio_feats(run: Run, cid: str, split: str, recs):
    legacy = Path("data/_features_lavdf") / cid / f"{split}.npz"
    if split in ("train", "val") and legacy.is_file():     # the features of scripts/i6/train_audio_lavdf.py (same window, same encoder)
        z = np.load(legacy)
        pos = {str(s): i for i, s in enumerate(z["sample_id"])}
        return z["feats"][[pos[r.sample_id] for r in recs]].astype(np.float32)
    d = K.audio_features(cid, "trim_norm", K.LAVDF, split, run.final, run.limit, run.root, modality="video", tag="lavdf")
    assert [str(s) for s in d["sample_id"]] == [r.sample_id for r in recs]
    return d["feats"].astype(np.float32)


def blocks_lavdf(run: Run, tag: str):
    splits = ["train", "val"] + (["test_in"] if run.final else [])
    recs = {s: K.select(K.LAVDF, s, run.final, "video", run.limit) for s in splits}
    meta = {s: lavdf_meta(recs[s]) for s in splits}
    if tag == "I-5":
        vis_model = lambda s: f"IMG-17_lr1e-05_s{s}"; vis_w = lambda s: Path(f"checkpoints/I-1/IMG-17_lr1e-05_s{s}/weights.pt")  # noqa: E731
        aud_head = lambda c, s: Path("data/_results/I-4") / f"head_{c}_s{s}.npz"  # noqa: E731
    elif tag == "I-6":
        vis_model = lambda s: f"IMG-17_lavdf_s{s}"; vis_w = lambda s: Path(f"checkpoints/I-6/IMG-17_lavdf_s{s}/weights.pt")  # noqa: E731
        aud_head = lambda c, s: Path("data/_results/I-6") / f"audio_head_{c}_s{s}.npz"  # noqa: E731
    else:   # I-6b
        vis_model = lambda s: f"IMG-17_lavdf_produced_s{s}"; vis_w = lambda s: Path(f"checkpoints/I-6/IMG-17_lavdf_produced_s{s}/weights.pt")  # noqa: E731
        aud_head = None
    # visual clip probabilities [seed][split]
    vis = {}
    for s in SEEDS:
        res = K.ft_scores(vis_model(s), "IMG-17", str(run.use(vis_w(s))), [(K.LAVDF, sp, recs[sp]) for sp in splits], run.root)
        for sp in splits:
            d = res[("lavdf", sp)]
            assert np.array_equal(d["sample_id"], meta[sp]["sample_id"])
            vis[(s, sp)] = d["p"]
    aud = {}
    if aud_head is not None:
        for c in ENSEMBLE:
            feats = {sp: lavdf_audio_feats(run, c, sp, recs[sp]) for sp in splits}
            for s in SEEDS:
                h = NpHead(run.use(aud_head(c, s)))
                for sp in splits:
                    aud[(c, s, sp)] = h.logit_rows(feats[sp])
        K.release()
    out = []

    def emit(modality, name, sp, s, p, label, mask=None):
        m = np.ones(len(p), bool) if mask is None else mask
        out.append(Block(modality, name, sp, s, meta[sp]["sample_id"][m], meta[sp]["group_id"][m], label[m], p[m], int(m.sum())))
    fused = {}
    if tag == "I-6":   # per seed: temperatures and the logistic regression on the fusion-fit split (train), E-011
        from analyse_i6 import fit_lr
        from dfd.eval import calibration as C
        for s in SEEDS:
            ens = {sp: np.mean([aud[(c, s, sp)] for c in ENSEMBLE], axis=0) for sp in splits}
            ok = ~np.isnan(vis[(s, "train")])
            Tv = C.fit_temperature(logit(vis[(s, "train")][ok]), meta["train"]["y_video"][ok]).temperature
            Ta = C.fit_temperature(ens["train"], meta["train"]["y_audio"]).temperature
            lv = {sp: np.nan_to_num(logit(vis[(s, sp)]) / Tv, nan=0.0) for sp in splits}
            la = {sp: ens[sp] / Ta for sp in splits}
            w = fit_lr(np.c_[lv["train"], la["train"]], meta["train"]["y_fused"])
            run.notes.append(f"fusion seed {s}: Tv {Tv:.4f}, Ta {Ta:.4f}, logistic regression intercept {w[0]:+.4f}, visual weight {w[1]:.4f}, audio weight {w[2]:.4f}")
            for sp in splits:
                fused[(s, sp)] = sigmoid(w[0] + w[1] * lv[sp] + w[2] * la[sp])
    for s in SEEDS:
        for sp in splits:
            if sp == "train":
                continue        # the fusion-fit split is a fitting set, not part of the score file
            m = meta[sp]
            pv = vis[(s, sp)]
            prod = m["cell"] != "real"
            emit("visual", "lavdf", sp, s, pv, m["y_fused"])
            emit("visual", "lavdf_videolabel", sp, s, pv, m["y_video"])
            emit("visual", "lavdf_produced_videolabel", sp, s, pv, m["y_video"], prod)
            if aud_head is not None:
                pa = sigmoid(np.mean([aud[(c, s, sp)] for c in ENSEMBLE], axis=0))
                emit("audio", "lavdf", sp, s, pa, m["y_fused"])
                emit("audio", "lavdf_audiolabel", sp, s, pa, m["y_audio"])
                emit("audio", "lavdf_produced_audiolabel", sp, s, pa, m["y_audio"], prod)
            if tag == "I-6":
                emit("fused", "lavdf", sp, s, fused[(s, sp)], m["y_fused"])
    return out


# ------------------------------------------------------------------ assembly
def blocks_for(run: Run, tag: str, task: str):
    out = []
    want = lambda t: task in ("all", t)   # noqa: E731
    if tag in ("I-0", "I-1", "I-2", "I-3", "I-4", "BASE"):
        if want("T-V"):
            if tag == "I-0":
                out += run.guarded("T-V frozen IMG-07", lambda: blocks_video_frozen(run))
            elif tag == "BASE":
                def base():
                    lr = json.loads(Path("data/_results/baseline/selection.json").read_text())["best_lr"]
                    return blocks_video_ft(run, "IMG-04", lambda s: f"IMG-04_lr{lr:g}_s{s}", lambda s: Path(f"checkpoints/I-1/IMG-04_lr{lr:g}_s{s}/weights.pt"))
                out += run.guarded("T-V Xception", base)
            else:
                out += run.guarded("T-V IMG-17 fine-tuned", lambda: blocks_video_ft(run, "IMG-17", lambda s: f"IMG-17_lr1e-05_s{s}",
                                                                                  lambda s: Path(f"checkpoints/I-1/IMG-17_lr1e-05_s{s}/weights.pt")))
            out += derived_celebdf(out)
        if tag != "BASE":
            if want("T-I"):
                out += run.guarded("T-I IMG-13", lambda: blocks_image(run))
            if want("T-A"):
                kind = {"I-0": "first", "I-1": "first", "I-2": "trim_norm_i2", "I-3": "trim_norm_i3", "I-4": "ensemble"}[tag]
                out += run.guarded("T-A " + kind, lambda: blocks_audio(run, kind))
    else:
        if want("LAV"):
            out += run.guarded("LAV-DF " + tag, lambda: blocks_lavdf(run, tag))
    return out


def out_tag(tag: str, b: Block) -> str:
    """The run pools all val rows of a run id for its threshold and temperature (dfd.eval.run), so score rows with different label schemes of LAV-DF
    (fused label, own modality label, own label within produced videos) go to separate iteration tags: I-5, I-5o, I-5p and the same for I-6, I-6b."""
    if b.dataset.startswith("lavdf_produced"):
        return tag + "p"
    if b.dataset in ("lavdf_videolabel", "lavdf_audiolabel"):
        return tag + "o"
    return tag


def write_csv(path: Path, tag: str, blocks):
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = {}
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(COLUMNS)
        for b in blocks:
            ok = np.isfinite(b.p)
            run_id = f"{tag}-{b.modality}-s{b.seed}"
            for sid, gid, lab, p in zip(b.sample_id[ok], b.group_id[ok], b.label[ok], b.p[ok]):
                w.writerow([run_id, tag, b.seed, b.modality, b.dataset, b.split, sid, gid, int(lab), f"{float(np.clip(p, 0.0, 1.0)):.8f}"])
            counts[f"{b.modality}|{b.dataset}|{b.split}|s{b.seed}"] = dict(rows=int(ok.sum()), listed=int(b.listed), no_score=int((~ok).sum()))
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iteration", nargs="+", required=True, help="one or more of " + ", ".join(ITERATIONS) + ", or all")
    ap.add_argument("--final", action="store_true", help="the real run: reads the sealed splits (the freeze); needs a clean decision to run")
    ap.add_argument("--dry-run", action="store_true", help="smoke test: val, dev_x and the LAV-DF fusion-fit split only; output under data/_final_dry")
    ap.add_argument("--limit", type=int, default=None, help="dry run only: an evenly strided subset of N items per split")
    ap.add_argument("--task", default="all", choices=["all", "T-V", "T-I", "T-A", "LAV"])
    a = ap.parse_args()
    if a.final == a.dry_run:
        raise SystemExit("refused: pass exactly one of --final (sealed splits are read) and --dry-run (they are not)")
    if a.final and a.limit:
        raise SystemExit("refused: --limit is for dry runs only")
    tags = ITERATIONS if a.iteration == ["all"] else a.iteration
    bad = [t for t in tags if t not in ITERATIONS]
    if bad:
        raise SystemExit(f"unknown iteration {bad}")
    for tag in tags:
        run = Run(a)
        t0 = time.time()
        print(f"== {tag} ({'FINAL' if run.final else 'dry run'}), task {a.task}", flush=True)
        blocks = blocks_for(run, tag, a.task)
        if not blocks:
            print(f"{tag}: no blocks produced", flush=True)
            continue
        suffix = "" if a.task == "all" else f".{a.task}.partial"
        groups: dict[str, list] = {}
        for b in blocks:
            groups.setdefault(out_tag(tag, b), []).append(b)
        for otag, bl in groups.items():
            path = run.root / "scores" / f"{otag}{suffix}.csv"
            counts = write_csv(path, otag, bl)
            prov = dict(iteration=otag, base_iteration=tag, task=a.task, mode="final" if run.final else "dry-run", limit=run.limit, date=time.strftime("%Y-%m-%d %H:%M:%S"),
                        git_commit=K.git_commit(), seconds=round(time.time() - t0), score_file=str(path), score_file_sha256=K.sha256(path),
                        files_used_sha256=run.files, split_files={d["key"]: [d["split_csv"], K.sha256(d["split_csv"])] for d in K.VIDEO_SETS + K.IMAGE_SETS + K.AUDIO_SETS + [K.LAVDF]},
                        counts=counts, notes=run.notes)
            path.with_suffix(".provenance.json").write_text(json.dumps(prov, indent=1))
            print(f"{otag}: {sum(c['rows'] for c in counts.values())} rows in {len(counts)} blocks -> {path}", flush=True)
        K.release()


if __name__ == "__main__":
    main()
