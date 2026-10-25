"""E-019 v2: canonical preprocessing protocol v2 (v1 plus setparams); see docs/EXPERIMENTS.md, E-019 v2 addendum.
The single definition of the ffmpeg command is dfd.platform.genvideo.canonical.ffmpeg_args.

Run: PYTHONPATH=src CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe scripts/canonical/canonicalise_v2.py
Outputs: data/_canonical_v2/<corpus>/<same relative path> (git-ignored), data/_results/canonical_v2/*.json
"""
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dfd.platform.genvideo import canonical as can

ROOT = Path(__file__).resolve().parents[2]
OUTR = ROOT / "data" / "_results" / "canonical_v2"
CORPORA = {"deepaction": ROOT / "data" / "deepaction_v1", "df26": ROOT / "data" / "df26"}
FFMPEG_VERSION = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0]


def files_for(name, base):
    if name == "deepaction":
        return sorted(p for p in base.glob("*/*/*.mp4") if p.parent.parent.name != "frames")
    return sorted(list((base / "fake").rglob("*.mp4")) + list((base / "real").rglob("*.mp4")))


def duration(p):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return None


def encode(args):
    src, dst = args
    if dst.exists() and dst.stat().st_size > 0:
        return src, "skip", None
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".tmp.mp4")
    r = subprocess.run(can.ffmpeg_args(src, tmp), capture_output=True, text=True)
    if r.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
        return src, "fail", r.stderr[-300:]
    tmp.replace(dst)
    return src, "ok", None


def main():
    OUTR.mkdir(parents=True, exist_ok=True)
    summary = dict(protocol=can.PROTOCOL_ID, ffmpeg=FFMPEG_VERSION, vf=can.VF,
                   example_args=can.ffmpeg_args("IN", "OUT"))
    for name, base in CORPORA.items():
        files = files_for(name, base)
        out_base = ROOT / "data" / "_canonical_v2" / name
        jobs = [(f, out_base / f.relative_to(base)) for f in files]
        with ThreadPoolExecutor(max_workers=8) as ex:
            res = list(ex.map(encode, jobs))
            durs = list(ex.map(duration, files))
        fails = [dict(path=str(s.relative_to(base)), err=e) for s, st, e in res if st == "fail"]
        short = [dict(path=str(f.relative_to(base)), duration=d) for f, d in zip(files, durs)
                 if d is not None and d < 2.0]
        unknown = [str(f.relative_to(base)) for f, d in zip(files, durs) if d is None]
        summary[name] = dict(n_source=len(files), ok=sum(st == "ok" for _, st, _ in res),
                             skipped=sum(st == "skip" for _, st, _ in res), failed=len(fails),
                             n_shorter_than_2s=len(short), n_unknown_duration=len(unknown))
        (OUTR / f"{name}_failures.json").write_text(json.dumps(fails, indent=1), encoding="utf-8")
        (OUTR / f"{name}_short_clips.json").write_text(json.dumps(short, indent=1), encoding="utf-8")
        print(name, summary[name], flush=True)
    (OUTR / "canonicalise_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
