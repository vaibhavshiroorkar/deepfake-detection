"""Cache frozen AV-HuBERT Base features of the T-AV windows (E-036). Train and val only; test_in needs --final.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/av/extract_features.py train
    PYTHONPATH=src .venv/Scripts/python.exe scripts/av/extract_features.py val

Configuration fixed by E-035 part A: Base, last layer, eye_px 64. Output: data/_features_av/<split>/<sha256 prefix>/<sha256>.npz
(fused, lips, audio as float16 [50, 768], face share) and data/_features_av/<split>/index.csv (sample_id, file, face, status).
Resumable: a window whose .npz exists is skipped.

For val, every window that is not forged also gets `fused_shuffled`: the fused pass of its lips with the audio of a donor
window from another group, by the derangement rule of E-035 (default_rng(20261102) permutation of the clean windows, each
taking the next one's audio; a same-group pair takes the first window of another group in permutation order). The donor's
audio-only features are in the donor's own file, so the I-8b and I-8c controls need nothing else.
"""
from __future__ import annotations

import argparse
import csv
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path("data")
MANIFEST = ROOT / "_manifests" / "av_lavdf_windows.manifest.csv"
SPLIT = ROOT / "_splits" / "av_lavdf_windows-8c8203b7d252.csv"
WEIGHTS = ROOT / "downloads" / "avhubert" / "base_vox_iter5.pt"
LANDMARKER = ROOT / "downloads" / "mediapipe" / "face_landmarker.task"
OUT = ROOT / "_features_av"
EYE_PX = 64
SEED = 20261102
WORKERS = 5


def rows_of(split: str) -> list[dict]:
    assignment = {r["sample_id"]: r["split"] for r in csv.DictReader(open(SPLIT, encoding="utf-8"))}
    out = []
    for r in csv.DictReader(open(MANIFEST, encoding="utf-8")):
        if assignment[r["sample_id"]] == split:
            notes = dict(kv.split("=", 1) for kv in r["notes"].split(";"))
            out.append({**r, "start": float(notes["start"]), "kind": notes["kind"], "cell": notes["cell"]})
    return out


def donors(clean: list[dict]) -> dict[int, int]:
    """The E-035 derangement over the clean (genuine and original) windows, by index into `clean`."""
    order = np.random.default_rng(SEED).permutation(len(clean))
    out = {}
    for i, j in zip(order, np.roll(order, 1), strict=True):
        if clean[i]["group_id"] == clean[j]["group_id"]:
            j = next(k for k in order if clean[k]["group_id"] != clean[i]["group_id"])
        out[int(i)] = int(j)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=("train", "val", "test_in"))
    parser.add_argument("--final", action="store_true", help="required for test_in (CLAUDE.md rule 3)")
    args = parser.parse_args()
    if args.split == "test_in" and not args.final:
        sys.exit("test_in is sealed: extract it only for a frozen T-AV iteration, with --final")

    from dfd.av.avhubert import load_avhubert
    from dfd.av.features import cache_path, encode, init_worker, prepare_window

    rows = rows_of(args.split)
    root = OUT / args.split
    index_path = root / "index.csv"
    done = {}
    if index_path.exists():
        done = {r["sample_id"]: r for r in csv.DictReader(open(index_path, encoding="utf-8"))}
    model = load_avhubert(WEIGHTS).to("cuda")
    keep_inputs = args.split == "val"
    inputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    todo = [r for r in rows if r["sample_id"] not in done or keep_inputs]
    jobs = [(str(ROOT / r["path"]), r["start"]) for r in todo]
    root.mkdir(parents=True, exist_ok=True)
    new = open(index_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(new, fieldnames=["sample_id", "file", "face", "status"])
    if not done:
        writer.writeheader()
    with Pool(WORKERS, initializer=init_worker, initargs=(str(LANDMARKER), EYE_PX)) as pool:
        for count, (r, (face, crops, audio)) in enumerate(zip(todo, pool.imap(prepare_window, jobs, chunksize=2), strict=True), 1):
            sid = r["sample_id"]
            path = cache_path(root, sid)
            if crops is None:
                status = face if isinstance(face, str) else "no face"
                face = 0.0 if isinstance(face, str) else face
            else:
                status = "ok"
                # The same clean set as E-035 part B (face in at least 80 percent of frames), so the donors match.
                if keep_inputs and r["kind"] != "forged" and face >= 0.8:
                    inputs[sid] = (crops, audio)
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(path, face=np.float32(face), **encode(model, crops, audio))
            if sid not in done:
                writer.writerow({"sample_id": sid, "file": str(path.relative_to(root)).replace("\\", "/"),
                                 "face": round(float(face), 4), "status": status})
            if count % 500 == 0:
                new.flush()
                print(f"{args.split}: {count} of {len(todo)}", flush=True)
    new.close()

    if keep_inputs:
        import torch

        from dfd.av.mouth import normalise_lips
        from dfd.av.sync import model_inputs

        clean = [r for r in rows if r["sample_id"] in inputs]
        for i, j in donors(clean).items():
            sid, donor = clean[i]["sample_id"], clean[j]["sample_id"]
            crops, _ = inputs[sid]
            video, sound = model_inputs(normalise_lips(crops), inputs[donor][1], "cuda")
            with torch.no_grad():
                shuffled = model(video=video, audio=sound)[0].half().cpu().numpy()
            path = cache_path(root, sid)
            stored = dict(np.load(path))
            np.savez(path, **stored, fused_shuffled=shuffled, donor=np.array(donor))
        print(f"val: shuffled fused pass written for {len(clean)} clean windows")


if __name__ == "__main__":
    main()
