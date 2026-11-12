"""E-035: zero-shot AV-HuBERT lip-audio agreement on LAV-DF windows (I-8a). Preregistered in docs/EXPERIMENTS.md.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/av/e035.py part-a
    PYTHONPATH=src .venv/Scripts/python.exe scripts/av/e035.py part-b

part-a  crop and model check on 300 original-real TRAIN windows; writes part_a.csv and part_a.json (the chosen configuration).
part-b  the shuffled-audio control first, then I-8a on the VAL windows with part A's configuration; writes scores_val.csv,
        control.json and, only if the control passes, results.json.

Decoding and MediaPipe run in worker processes; AV-HuBERT runs in the main process on the GPU. The split's test_in rows are
never read: the loader takes only the rows of the named split.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path("data")
OUT = ROOT / "_results" / "av" / "E-035"
MANIFEST = ROOT / "_manifests" / "av_lavdf_windows.manifest.csv"
SPLIT = ROOT / "_splits" / "av_lavdf_windows-8c8203b7d252.csv"
WEIGHTS = {"base": ROOT / "downloads" / "avhubert" / "base_vox_iter5.pt", "large": ROOT / "downloads" / "avhubert" / "large_vox_iter5.pt"}
MIDDLE = {"base": 6, "large": 12}
EYE_PX = (48, 56, 64, 72, 80)
SEED = 20261102
N_PART_A = 300
SHIFT_S = 0.6
MIN_FACE = 0.8
WORKERS = 5

_cropper = None


def rows_of(split: str) -> list[dict]:
    assignment = {r["sample_id"]: r["split"] for r in csv.DictReader(open(SPLIT, encoding="utf-8"))}
    out = []
    for r in csv.DictReader(open(MANIFEST, encoding="utf-8")):
        if assignment[r["sample_id"]] != split:
            continue
        notes = dict(kv.split("=", 1) for kv in r["notes"].split(";"))
        out.append({**r, "start": float(notes["start"]), "kind": notes["kind"], "cell": notes["cell"]})
    return out


def _init() -> None:
    global _cropper
    from dfd.av.mouth import MouthCropper

    _cropper = MouthCropper(ROOT / "downloads" / "mediapipe" / "face_landmarker.task")


def _prepare(job: tuple):
    """Decode one window and cut grey 96 px mouth crops at every eye_px. Runs in a worker.

    `job` is (path, start, seconds, eyes) or, with a fifth item True, the lips are the LAST 2 s of the read span.
    """
    from dfd.av.mouth import crop_mouth, read_window

    path, start, seconds, eyes, *tail = job
    try:
        frames, audio = read_window(ROOT / path, start, seconds)
        frames = frames[-50:] if tail and tail[0] else frames[:50]
        poses = _cropper.poses(frames)
        found = np.array([p is not None for p in poses])
        if not found.any():
            return found.mean() if len(found) else 0.0, None, audio
        have = np.flatnonzero(found)
        nearest = [poses[have[np.abs(have - i).argmin()]] for i in range(len(frames))]
        crops = {e: np.stack([crop_mouth(f, p, e) for f, p in zip(frames, nearest, strict=True)]) for e in eyes}
        return float(found.mean()), crops, audio
    except Exception as error:  # a broken file is counted, not fatal
        return f"{type(error).__name__}: {str(error)[:120]}", None, None


def _pool_map(jobs):
    with Pool(WORKERS, initializer=_init) as pool:
        yield from pool.imap(_prepare, jobs, chunksize=2)


def _models(names):
    import torch

    from dfd.av.avhubert import load_avhubert

    return {n: load_avhubert(WEIGHTS[n]).to("cuda") for n in names}, torch


def _agree(model, lips_u8, audio, layer):
    from dfd.av.mouth import normalise_lips
    from dfd.av.sync import agreement, model_inputs

    video, sound = model_inputs(normalise_lips(lips_u8), audio, "cuda")
    return agreement(model, video, sound, layer=layer)[0].mean().item()


def _group_bootstrap_mean(values: np.ndarray, groups: list[str], rng: np.random.Generator, n: int = 2000) -> tuple[float, float]:
    names = sorted(set(groups))
    index = {g: [i for i, h in enumerate(groups) if h == g] for g in names}
    reps = []
    for _ in range(n):
        pick = rng.integers(0, len(names), len(names))
        reps.append(np.mean(np.concatenate([values[index[names[k]]] for k in pick])))
    return float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


def part_a() -> None:
    rows = [r for r in rows_of("train") if r["kind"] == "original"]
    rng = np.random.default_rng(SEED)
    rows = [rows[i] for i in sorted(rng.choice(len(rows), N_PART_A, replace=False))]
    models, torch = _models(("base", "large"))
    configs = [(m, layer, e) for m in ("base", "large") for layer in (None, MIDDLE[m]) for e in EYE_PX]
    table = []
    duration = {m["file"]: m["duration"] for m in json.loads((ROOT / "lavdf" / "LAV-DF" / "metadata.min.json").read_text())}
    # Shift forward when the clip has room, else backward, so the shifted audio is never zero-padded.
    forward = [r["start"] + 2.0 + SHIFT_S <= duration[r["path"].removeprefix("lavdf/LAV-DF/")] for r in rows]
    short = {r["sample_id"] for r, f in zip(rows, forward, strict=True) if not f and r["start"] < SHIFT_S}
    excluded = [{"sample_id": s, "reason": "clip too short for a 0.6 s shift either way"} for s in sorted(short)]
    keep = [i for i, r in enumerate(rows) if r["sample_id"] not in short]
    rows, forward = [rows[i] for i in keep], [forward[i] for i in keep]
    jobs = [(r["path"], r["start"] if f else r["start"] - SHIFT_S, 2.0 + SHIFT_S, EYE_PX, not f) for r, f in zip(rows, forward, strict=True)]
    for count, (r, f, (coverage, crops, audio)) in enumerate(zip(rows, forward, _pool_map(jobs), strict=True), start=1):
        if crops is None or coverage < MIN_FACE:
            excluded.append({"sample_id": r["sample_id"], "reason": coverage if isinstance(coverage, str) else f"face {coverage:.2f}"})
            continue
        lag = int(SHIFT_S * 16000)
        if f:
            aligned, shifted = audio[:32000], audio[lag : lag + 32000]
        else:
            aligned, shifted = audio[lag : lag + 32000], audio[:32000]
        out = {"sample_id": r["sample_id"], "group_id": r["group_id"], "face": round(coverage, 3)}
        with torch.no_grad():
            for m, layer, e in configs:
                key = f"{m}|{layer or 'last'}|{e}"
                a, s = _agree(models[m], crops[e], aligned, layer), _agree(models[m], crops[e], shifted, layer)
                out[key] = int(a > s)
                out[key + "|margin"] = round(a - s, 5)
        table.append(out)
        if count % 25 == 0:
            print(f"part A {count} of {len(rows)}", flush=True)
    keys = [f"{m}|{layer or 'last'}|{e}" for m, layer, e in configs]
    with open(OUT / "part_a.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    groups = [t["group_id"] for t in table]
    accuracy = {k: float(np.mean([t[k] for t in table])) for k in keys}
    leader = max(keys, key=lambda k: accuracy[k])
    lead = np.array([t[leader] for t in table], float)
    ties, intervals = [], {}
    for k in keys:
        diff = np.array([t[k] for t in table], float) - lead
        low, high = _group_bootstrap_mean(diff, groups, np.random.default_rng(SEED))
        intervals[k] = [round(low, 4), round(high, 4)]
        if k == leader or low <= 0.0 <= high:
            ties.append(k)

    def cost(k):
        m, layer, e = k.split("|")
        return (m != "base", layer != "last", abs(int(e) - 64))

    chosen = min(ties, key=cost)
    m, layer, e = chosen.split("|")
    result = {"windows": len(table), "excluded": excluded, "accuracy": accuracy, "leader": leader,
              "difference_to_leader_95": intervals, "tied_with_leader": ties, "chosen": chosen,
              "config": {"model": m, "layer": None if layer == "last" else int(layer), "eye_px": int(e)}}
    (OUT / "part_a.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ("windows", "leader", "chosen")}, indent=1))
    print("excluded", len(excluded))


def _auc(y, s, groups):
    from dfd.eval.bootstrap import cluster_bootstrap

    r = cluster_bootstrap(np.asarray(y), np.asarray(s), np.asarray(groups), "auc", n_boot=2000, seed=SEED)
    return {"auc": round(r.estimate, 4), "low": round(r.low, 4), "high": round(r.high, 4), "n": len(y),
            "positives": int(np.sum(y)), "groups": len(set(groups))}


def part_b() -> None:
    from dfd.av.mouth import normalise_lips
    from dfd.av.sync import agreement, best_offset, model_inputs

    config = json.loads((OUT / "part_a.json").read_text())["config"]
    rows = rows_of("val")
    models, torch = _models((config["model"],))
    model = models[config["model"]]
    eye, layer = config["eye_px"], config["layer"]
    kept, excluded, audios, lips_of = [], [], {}, {}
    jobs = [(r["path"], r["start"], 2.0, (eye,)) for r in rows]
    for count, (r, (coverage, crops, audio)) in enumerate(zip(rows, _pool_map(jobs), strict=True), start=1):
        if crops is None or coverage < MIN_FACE:
            excluded.append({"sample_id": r["sample_id"], "cell": r["cell"], "kind": r["kind"],
                             "reason": coverage if isinstance(coverage, str) else f"face {coverage:.2f}"})
            continue
        lips = normalise_lips(crops[eye])
        video, sound = model_inputs(lips, audio, "cuda")
        with torch.no_grad():
            s = -agreement(model, video, sound, layer=layer)[0].mean().item()
            s_off = -best_offset(model, video, sound, layer=layer)[0]
        kept.append({"sample_id": r["sample_id"], "group_id": r["group_id"], "label": int(r["label"]), "kind": r["kind"],
                     "cell": r["cell"], "face": round(coverage, 3), "s": round(s, 6), "s_off": round(s_off, 6)})
        if r["kind"] != "forged":
            audios[r["sample_id"]] = audio
            lips_of[r["sample_id"]] = lips
        if count % 250 == 0:
            print(f"part B {count} of {len(rows)}", flush=True)
    with open(OUT / "scores_val.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(kept[0]))
        w.writeheader()
        w.writerows(kept)

    # Control: aligned windows against the same lips with another group's audio.
    clean = [k for k in kept if k["kind"] != "forged"]
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(clean))
    donor = {}
    for i, j in zip(order, np.roll(order, 1), strict=True):
        if clean[i]["group_id"] == clean[j]["group_id"]:
            j = next(k for k in order if clean[k]["group_id"] != clean[i]["group_id"])
        donor[i] = j
    y, scores, groups = [], [], []
    for i, k in enumerate(clean):
        video, sound = model_inputs(lips_of[k["sample_id"]], audios[clean[donor[i]]["sample_id"]], "cuda")
        with torch.no_grad():
            shuffled = -agreement(model, video, sound, layer=layer)[0].mean().item()
        y += [0, 1]
        scores += [k["s"], shuffled]
        groups += [k["group_id"], k["group_id"]]
    control = {**_auc(y, scores, groups), "line": 0.95}
    control["passed"] = control["auc"] >= control["line"]
    (OUT / "control.json").write_text(json.dumps(control, indent=1))
    print("control", control)
    if not control["passed"]:
        print("CONTROL FAILED: part B is not read (preregistered rule)")
        return

    results = {"config": config, "excluded": len(excluded),
               "excluded_by_cell": {c: sum(e["cell"] == c for e in excluded) for c in ("video_fake", "audio_fake", "both_fake", "real")},
               "face_coverage_by_cell": {c: round(float(np.mean([k["face"] for k in kept if k["cell"] == c])), 4)
                                         for c in ("video_fake", "audio_fake", "both_fake", "real")}}
    for score in ("s", "s_off"):
        block = {}
        produced = [k for k in kept if k["kind"] in ("forged", "genuine")]
        block["within_produced_all"] = _auc([k["label"] for k in produced], [k[score] for k in produced], [k["group_id"] for k in produced])
        for cell in ("video_fake", "audio_fake", "both_fake"):
            sub = [k for k in produced if k["cell"] == cell]
            block[f"within_produced_{cell}"] = _auc([k["label"] for k in sub], [k[score] for k in sub], [k["group_id"] for k in sub])
        clean_rows = [k for k in kept if k["kind"] in ("genuine", "original")]
        block["genuine_vs_original"] = _auc([int(k["kind"] == "genuine") for k in clean_rows], [k[score] for k in clean_rows],
                                            [k["group_id"] for k in clean_rows])
        block["all_val"] = _auc([k["label"] for k in kept], [k[score] for k in kept], [k["group_id"] for k in kept])
        results[score] = block
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("part", choices=("part-a", "part-b"))
    part = parser.parse_args().part
    if not (OUT / "prereg.json").exists():
        raise SystemExit("data/_results/av/E-035/prereg.json missing: run scripts/av/prereg.py E-035 first")
    os.environ.setdefault("GLOG_minloglevel", "2")
    {"part-a": part_a, "part-b": part_b}[part]()


if __name__ == "__main__":
    main()
