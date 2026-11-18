"""Score the v2 held-out sets with the website's current face and voice models and the candidates.

    .venv/Scripts/python.exe scripts/score_v2_eval.py [--device cuda] [--sets itw,asv,...]

Each item in results/v2-eval/sets.csv goes through the website's own path: a voice through
site.to_clip("voice"), a video as one frame at 1.0 s saved as JPEG and site.to_clip("picture"), then
the neutralizer and the preprocessor on the first 3-second part, as the site scores a first part. The
current model and every candidate score the same batch (Scorer.extra). Writes one line per item to
results/v2-eval/scores*.jsonl, appending, so a stopped run resumes where it stopped. With --shard i/n
a worker takes every n-th item from i and writes its own scores-i-of-n.jsonl (one writer per file). An item the website
would refuse (under 3 s of sound, no steady face) is written with its reason and no scores.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SETS = ROOT / "results" / "v2-eval" / "sets.csv"
SCORES = ROOT / "results" / "v2-eval" / "scores.jsonl"
PRETRAINED = ROOT / "models" / "pretrained"


def candidates(device: str, checks: set[str]) -> dict[str, dict]:
    """check -> {name: callable(batch) -> fake probability}, for every candidate whose files exist."""
    from deepfake_detection.webapp.candidates import (
        AntiDeepfakeVoice,
        ClipLnFace,
        EffortFace,
        OnBatch,
    )

    def voice(model):
        return OnBatch(model, "voice")

    def face(model):
        return OnBatch(model, "face")

    extra: dict[str, dict] = {"voice": {}, "face": {}}
    for name, folder in (
        ()
        if "voice" not in checks
        else (
            ("voice_adf_xlsr1b", "xls-r-1b-anti-deepfake"),
            ("voice_adf_mms300m", "mms-300m-anti-deepfake"),
        )
    ):
        if (PRETRAINED / folder / "model.safetensors").is_file():
            extra["voice"][name] = voice(
                AntiDeepfakeVoice(PRETRAINED / folder).to(device).eval()
            )
    clipln = PRETRAINED / "yermandy" / "model.torchscript"
    if "face" in checks and clipln.is_file():
        extra["face"]["face_clipln"] = face(ClipLnFace(clipln, device))
    effort = PRETRAINED / "effort" / "effort_ffpp.pth"
    clip = PRETRAINED / "clip-vit-large-patch14"
    if "face" in checks and effort.is_file() and (clip / "config.json").is_file():
        extra["face"]["face_effort"] = face(EffortFace(clip, effort).to(device).eval())
    return extra


def picture_of(video: Path, work: Path) -> Path:
    jpg = work / "picture.jpg"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-ss",
            "1.0",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            "-pix_fmt",
            "yuvj420p",
            str(jpg),
        ],
        check=True,
        timeout=120,
    )
    if not jpg.is_file():  # shorter than 1 s: take the first frame
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(video),
                "-frames:v",
                "1",
                "-q:v",
                "2",
                "-pix_fmt",
                "yuvj420p",
                str(jpg),
            ],
            check=True,
            timeout=120,
        )
    return jpg


def score_item(scorer, site, item: dict, work: Path) -> dict:
    from deepfake_detection.preprocessing.neutralize import (
        NeutralSettings,
        plan_windows,
    )
    from deepfake_detection.webapp.scorer import ClipRefused

    path = ROOT / item["path"]
    kind = item["kind"]
    try:
        source = picture_of(path, work) if kind == "picture" else path
        clip = site.to_clip(source, kind, work / "clip.mp4")
        start = 0.0
        if kind == "voice":
            parts = plan_windows(clip, NeutralSettings())
            if not parts:
                return {"refused": "under 3 s of sound"}
            start = parts[0][0]
        neutral = scorer._neutralize(clip, work / "neutral.mp4", start)
        checks, _ = scorer._face_and_voice(neutral, work / "cache", offset=start, top=1)
    except (
        ClipRefused,
        site.ClipRejected,
        RuntimeError,
        subprocess.SubprocessError,
    ) as error:
        return {"refused": f"{type(error).__name__}: {error}"[:200]}
    wanted = "voice" if kind == "voice" else "face"
    scores = {
        name: result.score for name, result in checks.items() if name.startswith(wanted)
    }
    reasons = {
        name: result.reason
        for name, result in checks.items()
        if name.startswith(wanted) and result.reason
    }
    return {"scores": scores, **({"reasons": reasons} if reasons else {})}


def main() -> int:
    from deepfake_detection.webapp import site
    from deepfake_detection.webapp.scorer import Scorer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--sets", default="", help="comma-separated set names; all when empty"
    )
    parser.add_argument(
        "--shard", default="", help="i/n: score every n-th item starting at i"
    )
    arguments = parser.parse_args()
    wanted = set(filter(None, arguments.sets.split(",")))
    with open(SETS, encoding="utf-8", newline="") as f:
        items = [r for r in csv.DictReader(f) if not wanted or r["set"] in wanted]
    out = SCORES
    if arguments.shard:
        index, count = (int(x) for x in arguments.shard.split("/"))
        items = items[index::count]
        out = SCORES.with_name(f"scores-{index}-of-{count}.jsonl")
    done = set()
    for path in SCORES.parent.glob("scores*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            done.add(json.loads(line)["path"])
    todo = [i for i in items if i["path"] not in done]
    print(f"{len(items)} items, {len(todo)} to score", flush=True)
    scorer = Scorer(device=arguments.device)
    scorer.extra = candidates(
        arguments.device, {"voice" if i["kind"] == "voice" else "face" for i in items}
    )
    scorer._explain_face = lambda *args, **kwargs: {}  # heatmaps are not scored here
    print("candidates:", {k: sorted(v) for k, v in scorer.extra.items()}, flush=True)
    began = time.time()
    for n, item in enumerate(todo, 1):
        with tempfile.TemporaryDirectory() as work:
            result = score_item(scorer, site, item, Path(work))
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps({**item, **result}) + "\n")
        if n % 25 == 0 or n == len(todo):
            rate = (time.time() - began) / n
            print(
                f"{n}/{len(todo)}  {rate:.1f} s/item  ~{rate * (len(todo) - n) / 60:.0f} min left",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
