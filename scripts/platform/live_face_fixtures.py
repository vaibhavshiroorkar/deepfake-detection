"""Fixture resolution for the face-bearing live end-to-end check. Datasets are referenced in place, never copied into the repo.

Only samples in a train or val split are used: ids are resolved with loaders.select_records(final=False), which refuses the test splits.
The LAV-DF clip is read from the official train directory (it is the clip already used by data/_results/freeze/web_freeze_check.json).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
SOURCES = [("ffpp.manifest.csv", "ffpp-037e2976fade.csv"), ("asvspoof2019_la.manifest.csv", "asvspoof2019_la-875bf337c379.csv")]
FFPP_FAKE, FFPP_REAL, ASV_FAKE = "ffpp:Deepfakes:084_042", "ffpp:original:084", "asvspoof2019_la:LA_D_1618135"
LAVDF_REL = "lavdf/LAV-DF/train/000470.mp4"


def non_test_records() -> dict:
    from dfd.data import loaders as L
    out = {}
    for man, spl in SOURCES:
        for sp in ("train", "val"):
            try:
                for r in L.select_records(ROOT / "data/_manifests" / man, ROOT / "data/_splits" / spl, sp):
                    out[r.sample_id] = (DATA / r.path, sp)
            except L.LoaderError:
                pass
    return out


def stored_scores() -> dict:
    par = json.loads((ROOT / "data/_results/freeze/parity.json").read_text())
    web = json.loads((ROOT / "data/_results/freeze/web_freeze_check.json").read_text())
    s = {c["sample_id"]: ("parity.json", c["stored_score"]) for c in par["cases"]}
    v = web["cases"]["video"]["per_modality"]
    s["lavdf:train/000470:visual"] = ("web_freeze_check.json", v["visual"]["score"])
    s["lavdf:train/000470:audio"] = ("web_freeze_check.json", v["audio"]["score"])
    return s


def frame_image(video: Path, out: Path, index: int = 40) -> Path:
    import cv2
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, fr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {index} of {video}")
    cv2.imwrite(str(out), fr)
    return out


def noface_video(out: Path) -> Path:
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-nostdin", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=3",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=3", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                    "-shortest", str(out)], check=True)
    return out


def build(workdir: Path) -> tuple[list[dict], list[str]]:
    """Return (fixtures, skip reasons). Each fixture: name, path, split, parity (list of (module_id, stored key))."""
    rec, fx, skip = non_test_records(), [], []

    def add(name, path, split, parity):
        fx.append({"name": name, "path": Path(path), "split": split, "parity": parity})

    for name, sid in [("ffpp_fake_video", FFPP_FAKE), ("ffpp_real_video", FFPP_REAL), ("asvspoof_fake_audio", ASV_FAKE)]:
        hit = rec.get(sid)
        if hit is None or not hit[0].is_file():
            skip.append(f"{name}: {sid} not in a train/val split or not on disk")
        else:
            add(name, hit[0], hit[1], [("speech" if name.startswith("asv") else "face_video", sid)])
    lav = DATA / LAVDF_REL
    if lav.is_file():
        add("lavdf_face_audio_video", lav, "official train directory",
            [("face_video", "lavdf:train/000470:visual"), ("speech", "lavdf:train/000470:audio")])
    else:
        skip.append(f"lavdf_face_audio_video: {lav} absent")
    fake = [f for f in fx if f["name"] == "ffpp_fake_video"]
    if fake:
        add("ffpp_fake_frame_image", frame_image(fake[0]["path"], workdir / "ffpp_084_042_frame40.png"), "val (frame of " + FFPP_FAKE + ")", [])
    add("synthetic_noface_video", noface_video(workdir / "noface.mp4"), "synthetic", [])
    return fx, skip
