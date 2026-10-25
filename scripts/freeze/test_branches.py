"""Readiness test of the extraction branches of the freeze harness (D-062, STEP 2). Real GPU, --limit sized, NO sealed split.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/freeze/test_branches.py [--limit 40]

The branches of video_features, image_features and audio_features that run on a sealed split (no sweep feature file exists: crops or waveforms are
read, the encoder runs, the file is cached) are forced on NON-sealed equivalents by switching off the LEGACY table: FF++ val and Celeb-DF dev_x
(video), ProGAN val and CNN-subset dev_x (image), ASVspoof dev and In-the-Wild dev_x (audio, both windows), LAV-DF val (audio of a video, tag lavdf).
Each result is compared with the stored sweep or training-time features of the same items (fp16 encoders are not bit-exact: reported as the maximum
absolute difference and the minimum cosine similarity per item). Output root data/_final_dry/branch_test (not part of any score file).
Also checks that the sealed guard refuses a sealed split without final=True, for each of the three branches.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402

ROOT = Path("data/_final_dry/branch_test")


def cos_min(a, b):
    a, b = a.astype(np.float32), b.astype(np.float32)
    num = (a * b).sum(-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12
    return float((num / den).min())


def ds_of(group, key):
    return next(d for d in group if d["key"] == key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40)
    a = ap.parse_args()
    legacy = dict(K.LEGACY)
    results = []
    t0 = time.time()

    def check(name, new, ref, mask=None):
        pos = {str(s): i for i, s in enumerate(ref["sample_id"])}
        idx = np.array([pos[str(s)] for s in new["sample_id"]])
        fn, fr = new["feats"], ref["feats"][idx]
        if mask is not None:   # video: compare the frames that exist in both
            m = new["mask"] & ref["mask"][idx]
            fn, fr = fn[m], fr[m]
        d = float(np.abs(fn.astype(np.float32) - fr.astype(np.float32)).max())
        c = cos_min(fn, fr)
        ok = c > 0.99 and np.array_equal(new["label"], ref["label"][idx])
        results.append((name, len(new["sample_id"]), d, c, ok))
        print(f"  {name}: {len(new['sample_id'])} items, max |diff| {d:.4g}, min cosine {c:.5f}, labels equal {np.array_equal(new['label'], ref['label'][idx])} -> {'OK' if ok else 'FAIL'}", flush=True)

    # sealed guard, all three branches
    for fn, args in ((K.video_features, ("IMG-07", ds_of(K.VIDEO_SETS, "celebdf_v2"), "test_x")), (K.image_features, ("IMG-13", ds_of(K.IMAGE_SETS, "diffusion_test"), "test_x")),
                     (K.audio_features, ("AU-8", "first", ds_of(K.AUDIO_SETS, "in_the_wild"), "test_x"))):
        try:
            fn(*args, False, 5, ROOT)
            results.append((f"guard {fn.__name__}", 0, 0, 0, False))
            print(f"  guard {fn.__name__}: NOT refused -> FAIL")
        except RuntimeError as e:
            results.append((f"guard {fn.__name__}", 0, 0, 0, True))
            print(f"  guard {fn.__name__}: refused ({str(e)[:60]}) -> OK", flush=True)

    K.LEGACY.clear()      # force the extraction branches
    try:
        print("video (FF++ val, Celeb-DF dev_x), IMG-07", flush=True)
        for key, split, stem in (("ffpp", "val", "ffpp_val"), ("celebdf_v2", "dev_x", "celebdf_dev_x")):
            new = K.video_features("IMG-07", ds_of(K.VIDEO_SETS, key), split, False, a.limit, ROOT)
            ref = dict(np.load(f"data/_features/IMG-07/T-V/{stem}.npz"))
            check(f"video {key}/{split}", new, ref, mask=True)
        print("image (ProGAN val, CNN-subset dev_x), IMG-13", flush=True)
        for key, split, stem in (("forensynths_progan_val", "val", "progan_val"), ("forensynths_cnn_subset", "dev_x", "cnn_dev_x")):
            new = K.image_features("IMG-13", ds_of(K.IMAGE_SETS, key), split, False, a.limit, ROOT)
            check(f"image {key}/{split}", new, dict(np.load(f"data/_features/IMG-13/T-I/{stem}.npz")))
        print("audio (ASVspoof dev, In-the-Wild dev_x), AU-8, windows first and trim_norm", flush=True)
        for variant, base in (("first", "data/_features"), ("trim_norm", "data/_features_trimnorm")):
            for key, split, stem in (("asvspoof2019_la", "val", "asvspoof_dev"), ("in_the_wild", "dev_x", "itw_dev_x")):
                new = K.audio_features("AU-8", variant, ds_of(K.AUDIO_SETS, key), split, False, a.limit, ROOT)
                check(f"audio {variant} {key}/{split}", new, dict(np.load(f"{base}/AU-8/T-A/{stem}.npz")))
        print("audio of a video (LAV-DF val, trim_norm, tag lavdf), AU-8", flush=True)
        new = K.audio_features("AU-8", "trim_norm", K.LAVDF, "val", False, a.limit, ROOT, modality="video", tag="lavdf")
        ref = dict(np.load("data/_features_lavdf/AU-8/val.npz"))
        check("audio lavdf/val", new, ref)
    finally:
        K.LEGACY.update(legacy)
        K.release()
    bad = [r for r in results if not r[4]]
    print(f"{len(results)} checks, {len(bad)} failed, {time.time() - t0:.0f} s")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
