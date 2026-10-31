"""Build the demonstration set: one file per manipulation type each detector covers, plus real files, in data/_demo_inputs/demo_set/.

Selection rule (docs/DEMO_SELECTION.md): for every category the eligible files are the non-sealed split named below, resolved with
loaders.select_records(final=False), which refuses test splits; they are sorted by sample id and ONE file is drawn with
random.Random(SEED + category index). No model is run and no score is read, so no file is chosen because the detector gets it right.
LAV-DF clips come from the official train directory and exclude every clip used to train the I-6 heads (lavdf_headtrain).
The folder is git-ignored; it is rebuilt by running this script again. A manifest.json next to the files records each sample id,
split, label, manipulation type, source path and SHA-256.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/platform/make_demo_set.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = DATA / "_demo_inputs" / "demo_set"
SEED = 20261031
M, S = DATA / "_manifests", DATA / "_splits"
FFPP = (M / "ffpp.manifest.csv", S / "ffpp-037e2976fade.csv", "val")
ASV = (M / "asvspoof2019_la.manifest.csv", S / "asvspoof2019_la-875bf337c379.csv", "val")
ITW = (M / "in_the_wild.manifest.csv", S / "in_the_wild-5cf2ea54a724.csv", "dev_x")
CDF = (M / "celebdf_v2.manifest.csv", S / "celebdf_v2-82a3a70594e9.csv", "dev_x")
CNN = (M / "forensynths_cnn_subset.manifest.csv", S / "cnn_subset-4e973bc4da0f.csv", "dev_x")
PGN = (M / "forensynths_progan_val.manifest.csv", S / "progan_val-b42dc9619141.csv", "val")

# (folder, file stem, label, manipulation type, source, filter on the record)
CATEGORIES = [
    ("face_video", "real_original_ffpp", 0, "real (unmanipulated)", FFPP, lambda r: r.method == "real"),
    ("face_video", "fake_deepfakes_faceswap_ffpp", 1, "face swap, autoencoder (Deepfakes)", FFPP, lambda r: r.method == "Deepfakes"),
    ("face_video", "fake_faceswap_graphics_ffpp", 1, "face swap, graphics-based (FaceSwap)", FFPP, lambda r: r.method == "FaceSwap"),
    ("face_video", "fake_face2face_reenactment_ffpp", 1, "expression reenactment (Face2Face)", FFPP, lambda r: r.method == "Face2Face"),
    ("face_video", "fake_neuraltextures_reenactment_ffpp", 1, "mouth reenactment, neural rendering (NeuralTextures)", FFPP,
     lambda r: r.method == "NeuralTextures"),
    ("face_video", "real_celebdf", 0, "real (unmanipulated)", CDF, lambda r: r.method == "real"),
    ("face_video", "fake_faceswap_celebdf", 1, "face swap, improved synthesis (Celeb-DF v2)", CDF, lambda r: r.method != "real"),
    ("speech", "real_bonafide_asvspoof", 0, "real (bona fide speech)", ASV, lambda r: r.method == "real"),
    ("speech", "fake_tts_A01_asvspoof", 1, "text to speech (ASVspoof A01)", ASV, lambda r: r.method == "A01"),
    ("speech", "fake_tts_A04_asvspoof", 1, "text to speech, waveform concatenation (ASVspoof A04)", ASV, lambda r: r.method == "A04"),
    ("speech", "fake_vc_A05_asvspoof", 1, "voice conversion (ASVspoof A05)", ASV, lambda r: r.method == "A05"),
    ("speech", "fake_vc_A06_asvspoof", 1, "voice conversion, spectral filtering (ASVspoof A06)", ASV, lambda r: r.method == "A06"),
    ("speech", "real_inthewild", 0, "real (public speech recording)", ITW, lambda r: r.method == "real"),
    ("speech", "fake_inthewild", 1, "speech deepfake found online (In-the-Wild)", ITW, lambda r: r.method != "real"),
    ("image", "real_progan_val", 0, "real photograph", PGN, lambda r: r.method == "real"),
    ("image", "fake_progan_val", 1, "GAN image (ProGAN, the training generator)", PGN, lambda r: r.method != "real"),
    ("image", "real_face_photo_cnn", 0, "real face photograph", CNN, lambda r: r.method == "real"),
]
# the six generators of the sealed CNN test (BigGAN, CycleGAN, ...) are test_x only; the development slice holds these two
CNN_GENERATORS = (("stargan", "face attribute editing GAN (StarGAN), unseen generator"), ("whichfaceisreal", "generated face (StyleGAN faces of whichfaceisreal.com), unseen generator"))
LAVDF_CELLS = [("real", "real (unmanipulated)", 0, 0), ("fake_video_only", "lip region re-rendered, audio untouched", 1, 0),
               ("fake_audio_only", "words replaced in the audio, video untouched", 0, 1), ("fake_audio_and_video", "audio and lips both edited", 1, 1)]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from dfd.data import loaders as L
    cache, rows, skipped = {}, [], []

    def pool(src):
        if src not in cache:
            cache[src] = sorted(L.select_records(src[0], src[1], src[2], final=False), key=lambda r: r.sample_id)
        return cache[src]

    cats = list(CATEGORIES) + [("image", f"fake_{g}_cnn", 1, kind, CNN,
                                (lambda g: lambda r: r.method != "real" and r.sample_id.split(":", 1)[1].split("/")[0] == g)(g))
                               for g, kind in CNN_GENERATORS]
    if OUT.exists():
        shutil.rmtree(OUT)
    for i, (folder, stem, label, kind, src, keep) in enumerate(cats):
        elig = [r for r in pool(src) if keep(r) and (DATA / r.path).is_file()]
        if not elig:
            skipped.append(f"{stem}: no eligible file on disk")
            continue
        r = random.Random(SEED + i).choice(elig)
        dst = OUT / folder / f"{len(rows) + 1:02d}_{stem}{Path(r.path).suffix.lower()}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(DATA / r.path, dst)
        rows.append({"file": dst.relative_to(OUT).as_posix(), "label": "fake" if label else "real", "manipulation": kind,
                     "detector": folder, "sample_id": r.sample_id, "split": src[2], "eligible_pool": len(elig),
                     "source_path": r.path, "sha256": sha256(dst)})

    meta = json.load(open(DATA / "lavdf" / "LAV-DF" / "metadata.min.json", encoding="utf-8"))
    used = {row["sample_id"] for row in csv.DictReader(open(M / "lavdf_headtrain.manifest.csv", encoding="utf-8"))}
    train = sorted((m for m in meta if m["split"] == "train" and "lavdf:" + m["file"] not in used), key=lambda m: m["file"])
    for j, (stem, kind, mv, ma) in enumerate(LAVDF_CELLS):
        elig = [m for m in train if int(m["modify_video"]) == mv and int(m["modify_audio"]) == ma and (DATA / "lavdf/LAV-DF" / m["file"]).is_file()]
        if not elig:
            skipped.append(f"lavdf {stem}: no eligible clip on disk")
            continue
        m = random.Random(SEED + 100 + j).choice(elig)
        dst = OUT / "audio_visual" / f"{len(rows) + 1:02d}_{stem}_lavdf_train.mp4"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(DATA / "lavdf/LAV-DF" / m["file"], dst)
        rows.append({"file": dst.relative_to(OUT).as_posix(), "label": "real" if not (mv or ma) else "fake", "manipulation": kind,
                     "detector": "audio_visual", "sample_id": "lavdf:" + m["file"], "split": "official train (not used by any head)",
                     "eligible_pool": len(elig), "source_path": "lavdf/LAV-DF/" + m["file"], "sha256": sha256(dst),
                     "label_video": "fake" if mv else "real", "label_audio": "fake" if ma else "real",
                     "fake_periods_s": m.get("fake_periods")})

    (OUT / "manifest.json").write_text(json.dumps({"seed": SEED, "rule": "one file per category drawn with random.Random(seed + index) "
                                                   "from the sorted non-sealed pool; no score read", "files": rows, "skipped": skipped},
                                                  indent=1) + "\n", encoding="utf-8")
    for row in rows:
        print(f"{row['file']:<58} {row['label']:<5} {row['split']:<8} {row['sample_id']}")
    for s in skipped:
        print("SKIPPED:", s)
    print(len(rows), "files written to", OUT)
    return 1 if skipped else 0


if __name__ == "__main__":
    sys.exit(main())
