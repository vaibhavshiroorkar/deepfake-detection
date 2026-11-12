"""E-019: canonical preprocessing protocol v1 (see docs/EXPERIMENTS.md, E-019).

Run: PYTHONPATH=src CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe scripts/canonical/canonicalise.py
Outputs: data/_canonical/<corpus>/<same relative path> (git-ignored), data/_results/canonical/*.json
"""
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTR = ROOT / "data" / "_results" / "canonical"

# ---- protocol constant block (do not edit; a change is a new protocol version)
PROTOCOL_VERSION = "v1"
FFMPEG_VERSION = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0]
FFMPEG_ARGS = ["-nostdin", "-map_metadata", "-1", "-map_chapters", "-1", "-an", "-sn", "-t", "2.0",
               "-vf", "fps=7,scale=224:224:force_original_aspect_ratio=increase:flags=area,crop=224:224",
               "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p", "-g", "14", "-bf", "2",
               "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
               "-x264-params", "no-info=1", "-movflags", "+faststart"]
# command: ffmpeg -nostdin -i IN <FFMPEG_ARGS minus -nostdin> OUT.mp4 (-y added for atomic temp overwrite)

CORPORA = {"deepaction": ROOT / "data" / "deepaction_v1", "df26": ROOT / "data" / "df26"}


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
    cmd = ["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(src)] + FFMPEG_ARGS[1:] + [str(tmp)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
        return src, "fail", r.stderr[-300:]
    tmp.replace(dst)
    return src, "ok", None


def main():
    OUTR.mkdir(parents=True, exist_ok=True)
    summary = dict(protocol=PROTOCOL_VERSION, ffmpeg=FFMPEG_VERSION, args=FFMPEG_ARGS)
    for name, base in CORPORA.items():
        files = files_for(name, base)
        out_base = ROOT / "data" / "_canonical" / name
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
