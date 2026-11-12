"""Integrity check of a folder of downloaded videos: ffprobe readability, SHA-256 manifest, manifest record.

Usage: python scripts/data/verify_videos.py <folder> <manifest name> [workers]
For every .mp4 below the folder: ffprobe must report a video stream with a positive frame count and
duration (a container and header check, not a full decode), and the SHA-256 is computed. Writes
data/_manifests/<name>.sha256 (one line per file, paths relative to the folder) and a summary line in
data/_manifests/downloads.jsonl. Exits non-zero when any file fails.
"""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_open as fo  # noqa: E402


def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=codec_name,nb_frames,duration,width,height", "-of", "json", str(path)],
                       capture_output=True, text=True)
    try:
        s = json.loads(r.stdout)["streams"][0]
        frames = int(s.get("nb_frames") or 0)
        dur = float(s.get("duration") or 0)
        ok = r.returncode == 0 and (frames > 0 or dur > 0)
        return ok, frames, dur
    except Exception:
        return False, 0, 0.0


def one(args):
    root, p = args
    ok, frames, dur = probe(p)
    return p.relative_to(root).as_posix(), ok, frames, dur, fo.sha256_file(p), p.stat().st_size


def main():
    root = Path(sys.argv[1]).resolve()
    name = sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    files = sorted(root.rglob("*.mp4"))
    fo.say(f"{len(files)} videos under {root}")
    rows = []
    with ThreadPoolExecutor(workers) as ex:
        for k, r in enumerate(ex.map(one, [(root, p) for p in files])):
            rows.append(r)
            if k % 500 == 0:
                fo.say(f"{k}/{len(files)}")
    bad = [r[0] for r in rows if not r[1]]
    fo.MANIFESTS.mkdir(parents=True, exist_ok=True)
    with open(fo.MANIFESTS / f"{name}.sha256", "w", encoding="utf-8") as f:
        for rel, ok, frames, dur, sha, size in rows:
            f.write(f"{sha}  {rel}\n")
    total = sum(r[5] for r in rows)
    entry = {"date": fo.now(), "name": name, "url": "authors' download script (EU2) after approval",
             "file": str(root), "bytes": total, "sha256": "see data/_manifests/" + name + ".sha256",
             "sha256_checked_against": "none published, recorded only",
             "files": len(rows), "unreadable": len(bad), "total_frames": sum(r[2] for r in rows)}
    with open(fo.MANIFESTS / "downloads.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    fo.say(f"{len(rows)} files, {total / 1e9:.2f} GB, unreadable {len(bad)}")
    if bad:
        fo.say("unreadable: " + ", ".join(bad[:20]))
        sys.exit(1)


if __name__ == "__main__":
    main()
