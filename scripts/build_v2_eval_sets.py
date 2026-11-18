"""Fixed held-out sets for choosing the website's v2 face and voice models.

    .venv/Scripts/python.exe scripts/build_v2_eval_sets.py

Every candidate is scored on the same items through the website's own path, so the sets are drawn once
with a fixed seed and written to results/v2-eval/sets.csv (one row per item: set, kind, label, path,
group). `kind` is the website mode the item runs in: "voice" or "picture".

Voice sets:
  itw      In-the-Wild, 12 speakers held out by a seeded draw of the 54 (written to the file, so a later
           voice model trained on In-the-Wild must leave them out), 200 real and 200 fake.
  asv      ASVspoof 2019 LA eval partition, 200 bona fide and 200 spoof.
  fav      FakeAVCeleb identity-strict test clips (the study's test list), 150 with real and 150 with
           fake audio, scored as voice only.
Face sets (one frame per video, as a picture):
  ffpp     FaceForensics++ c23 official test split: its 140 real videos and 40 from each of the five
           face methods (Deepfakes, Face2Face, FaceSwap, NeuralTextures, FaceShifter) on test pairs.
  celebdf  Celeb-DF v2 official test list, all 518 videos.
  favface  FakeAVCeleb identity-strict test clips, 150 with a real and 150 with a fake face.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results" / "v2-eval" / "sets.csv"
SEED = 19
FFPP_METHODS = ("deepfakes", "face2face", "faceswap", "neuraltextures", "faceshifter")
FAV_TEST = (
    ROOT / "results" / "p001-sync" / "sync-test-identity-strict-seed17-predictions.csv"
)


def _rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _sample(rows: list[dict], n: int, tag: str) -> list[dict]:
    rows = sorted(rows, key=lambda r: r["path"])
    return random.Random(f"{SEED}-{tag}").sample(rows, min(n, len(rows)))  # noqa: S311 - seeded draw


def itw() -> tuple[list[dict], list[str]]:
    rows = _rows(DATA / "manifests" / "in_the_wild.csv")
    speakers = sorted({r["speaker"] for r in rows})
    held = sorted(random.Random(f"{SEED}-itw-speakers").sample(speakers, 12))  # noqa: S311
    items = [
        {
            "set": "itw",
            "kind": "voice",
            "label": int(r["audio_fake"]),
            "path": r["audio_path"],
            "group": r["speaker"],
        }
        for r in rows
        if r["speaker"] in held
    ]
    out = []
    for label in (0, 1):
        out += _sample([i for i in items if i["label"] == label], 200, f"itw-{label}")
    return out, held


def asv() -> list[dict]:
    rows = [
        r
        for r in _rows(DATA / "manifests" / "asvspoof2019_la.csv")
        if r["partition"] == "eval"
    ]
    items = [
        {
            "set": "asv",
            "kind": "voice",
            "label": int(r["audio_fake"]),
            "path": r["audio_path"],
            "group": r["method"],
        }
        for r in rows
    ]
    out = []
    for label in (0, 1):
        out += _sample([i for i in items if i["label"] == label], 200, f"asv-{label}")
    return out


def _fav_test() -> list[dict]:
    test = {r["clip_id"] for r in _rows(FAV_TEST)}
    return [
        r for r in _rows(DATA / "manifests" / "fakeavceleb.csv") if r["clip_id"] in test
    ]


def fav(column: str, kind: str, name: str) -> list[dict]:
    items = [
        {
            "set": name,
            "kind": kind,
            "label": int(r[column]),
            "path": str(Path("data") / r["video_path"]),
            "group": r["source"],
        }
        for r in _fav_test()
    ]
    out = []
    for label in (0, 1):
        out += _sample(
            [i for i in items if i["label"] == label], 150, f"{name}-{label}"
        )
    return out


def ffpp() -> list[dict]:
    pairs = json.loads(
        (
            DATA / "legacy-team" / "manifests" / "ffpp_official_splits" / "test.json"
        ).read_text(encoding="utf-8")
    )
    names = {f"{a}_{b}" for a, b in pairs} | {f"{b}_{a}" for a, b in pairs}
    ids = {x for pair in pairs for x in pair}
    rows = _rows(DATA / "manifests" / "faceforensics.csv")
    out = []
    for r in rows:
        stem = Path(r["video_path"]).stem
        path = str(Path("data") / r["video_path"])
        if r["method"] == "real" and stem in ids:
            out.append(
                {
                    "set": "ffpp",
                    "kind": "picture",
                    "label": 0,
                    "path": path,
                    "group": "real",
                }
            )
    for method in FFPP_METHODS:
        fakes = [
            {
                "set": "ffpp",
                "kind": "picture",
                "label": 1,
                "path": str(Path("data") / r["video_path"]),
                "group": method,
            }
            for r in rows
            if r["method"] == f"ffpp-{method}" and Path(r["video_path"]).stem in names
        ]
        out += _sample(fakes, 40, f"ffpp-{method}")
    return out


def celebdf() -> list[dict]:
    out = []
    for line in (
        (DATA / "Celeb-DF-v2" / "List_of_testing_videos.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    ):
        if not line.strip():
            continue
        flag, rel = line.split(maxsplit=1)
        group = rel.split("/")[0]
        # In this list 1 marks a real video and 0 a synthesized one.
        out.append(
            {
                "set": "celebdf",
                "kind": "picture",
                "label": 1 - int(flag),
                "path": str(Path("data") / "Celeb-DF-v2" / rel),
                "group": group,
            }
        )
    return out


def main() -> int:
    itw_items, held = itw()
    items = (
        itw_items
        + asv()
        + fav("audio_fake", "voice", "fav")
        + ffpp()
        + celebdf()
        + fav("video_fake", "picture", "favface")
    )
    missing = [i["path"] for i in items if not (ROOT / i["path"]).is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} files missing, first: {missing[:3]}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["set", "kind", "label", "path", "group"])
        writer.writeheader()
        writer.writerows(items)
    (OUT.parent / "itw-held-out-speakers.json").write_text(
        json.dumps(held, indent=1), encoding="utf-8"
    )
    counts: dict = {}
    for i in items:
        counts.setdefault(i["set"], [0, 0])[i["label"]] += 1
    for name, (real, fake) in counts.items():
        print(f"{name}: {real} real, {fake} fake")
    print(f"-> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
