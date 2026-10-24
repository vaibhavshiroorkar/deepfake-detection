"""LAV-DF: resumable download, hash, extract, count, ffprobe every video, log, delete the tar.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/data/lavdf_job.py

Stages are recorded in data/_logs/lavdf_state.json, so a rerun resumes at the first unfinished stage. The token is read
by huggingface_hub from its default path and is never printed or written. The tar is deleted only when its SHA-256
equals the Hugging Face LFS value, extraction produced exactly the archive's file count and ffprobe found no
unreadable video; anything else keeps the tar and says why in the STATUS line and in DATA_LOG.md.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

os.environ.setdefault("HF_HUB_CACHE", r"D:\cache\hf\hub")
REPO, FILE, REV = "ControlNet/LAV-DF", "LAV-DF.tar", "c9158b3b9ad7f05eca712da18b8722d79d7f4d8e"
SIZE = 25607198720
TAR = Path("data/_incoming/lavdf") / FILE
OUT = Path("data/lavdf")
STATE = Path("data/_logs/lavdf_state.json")
VERIFY = Path("data/_logs/lavdf_verify.json")


def status_line(text: str) -> None:
    p = Path("docs/STATUS.md")
    t = p.read_text(encoding="utf-8")
    if "## Background job log" not in t:
        t = t.rstrip("\n") + "\n\n## Background job log\n\n"
    p.write_text(t.rstrip("\n") + f"\n- {date.today().isoformat()} {time.strftime('%H:%M')} {text}\n", encoding="utf-8", newline="")


def load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.is_file() else {}


def save_state(s: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(s, indent=1))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def download() -> None:
    from huggingface_hub import hf_hub_download
    TAR.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 200):
        try:
            hf_hub_download(REPO, FILE, repo_type="dataset", revision=REV, local_dir=str(TAR.parent))
            if TAR.stat().st_size == SIZE:
                return
        except Exception as e:  # network outage, retry (the partial file is resumed)
            print("download attempt", attempt, type(e).__name__, str(e)[:120], flush=True)
        time.sleep(min(120, 10 * attempt))
    raise RuntimeError("download did not finish")


def hf_sha() -> str:
    from huggingface_hub import HfApi
    info = HfApi().get_paths_info(REPO, [FILE], repo_type="dataset", revision=REV)[0]
    return info.lfs.sha256


def extract() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    n = collections.Counter()
    members = 0
    with tarfile.open(TAR, "r|") as tf:  # streamed, one pass
        for m in tf:
            tf.extract(m, OUT, filter="data")
            if m.isfile():
                members += 1
                n[Path(m.name).suffix.lower()] += 1
    files = sum(1 for p in OUT.rglob("*") if p.is_file())
    return dict(archive_files=members, extracted_files=files, by_suffix=dict(n))


def probe(path: Path) -> str:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return "error"
    return "audio" if r.stdout.strip() else "no_audio"


def ffprobe_all() -> dict:
    vids = sorted(OUT.rglob("*.mp4"))
    res = collections.Counter()
    bad = []
    with ThreadPoolExecutor(16) as ex:
        for p, r in zip(vids, ex.map(probe, vids)):
            res[r] += 1
            if r != "audio" and len(bad) < 20:
                bad.append(f"{p.relative_to(OUT).as_posix()}: {r}")
    return dict(videos=len(vids), **res, examples=bad)


def split_counts() -> dict:
    """Videos and identities per split from the metadata json shipped in the archive (keys are inspected, not assumed)."""
    metas = sorted(OUT.rglob("metadata*.json"))
    if not metas:
        return dict(note="no metadata json found in the archive")
    out = {"metadata_files": [m.relative_to(OUT).as_posix() for m in metas]}
    data = json.loads(metas[0].read_text(encoding="utf-8"))
    rows = data if isinstance(data, list) else data.get("data") or list(data.values())
    out["keys"] = sorted(rows[0].keys()) if rows and isinstance(rows[0], dict) else "NR"
    vids, ids = collections.defaultdict(int), collections.defaultdict(set)
    for r in rows:
        sp = r.get("split", "NR")
        vids[sp] += 1
        for k in ("original", "file", "source", "speaker", "identity"):
            m = re.search(r"id\d{5}", str(r.get(k, "")))
            if m:
                ids[sp].add(m.group(0))
                break
    out["videos_per_split"] = dict(vids)
    out["identities_per_split"] = {k: len(v) for k, v in ids.items()} or "NR (no idNNNNN token in the metadata)"
    return out


def main() -> int:
    s = load_state()
    if "downloaded" not in s:
        download()
        s["downloaded"] = time.strftime("%Y-%m-%d %H:%M")
        save_state(s)
        status_line("LAV-DF.tar downloaded (25.6 GB) to data/_incoming/lavdf/; hashing and extracting next.")
    if "sha" not in s:
        s["sha"] = sha256(TAR)
        s["sha_hf"] = hf_sha()
        s["sha_match"] = s["sha"] == s["sha_hf"]
        s["tar_bytes"] = TAR.stat().st_size
        save_state(s)
    if "extract" not in s:
        s["extract"] = extract()
        save_state(s)
    if "splits" not in s:
        s["splits"] = split_counts()
        save_state(s)
    if "ffprobe" not in s:
        s["ffprobe"] = ffprobe_all()
        save_state(s)
    VERIFY.write_text(json.dumps(s, indent=1))
    ex, fp = s["extract"], s["ffprobe"]
    ok = s["sha_match"] and ex["archive_files"] == ex["extracted_files"] and fp.get("error", 0) == 0
    deleted = False
    if ok and TAR.is_file():
        TAR.unlink()
        deleted = True
    block = f"""

## LAV-DF download and verification (job finished {time.strftime('%Y-%m-%d %H:%M')})

- Source: https://huggingface.co/datasets/ControlNet/LAV-DF at revision {REV}; file LAV-DF.tar, {s['tar_bytes']:,} bytes. SHA-256 computed locally: {s['sha']}; Hugging Face LFS value: {s['sha_hf']}; match: {s['sha_match']}.
- Extracted to data/lavdf/ with the Python tarfile "data" filter: {ex['archive_files']:,} files in the archive, {ex['extracted_files']:,} files on disk; by suffix {ex['by_suffix']}.
- Splits from the shipped metadata (README gives no counts; the 91 / 31 / 31 identities of D-021 are from the paper and are compared here): {json.dumps(s['splits'])}.
- ffprobe on every .mp4 (first audio stream): {fp['videos']:,} videos, {fp.get('audio', 0):,} with an audio stream, {fp.get('no_audio', 0):,} without, {fp.get('error', 0):,} unreadable. Examples of problems: {fp['examples'] or 'none'}.
- Tar deleted: {deleted}. {'' if deleted else 'It was kept because at least one check failed (hash, file count or unreadable videos); decided autonomously - needs user review.'}
- Terms: CC BY-NC 4.0 and the LAV-DF terms logged above (non-commercial research; VoxCeleb2 agreement applies).
"""
    p = Path("docs/DATA_LOG.md")
    p.write_text(p.read_text(encoding="utf-8").rstrip("\n") + block, encoding="utf-8", newline="")
    status_line(f"LAV-DF job finished: hash match {s['sha_match']}, {ex['extracted_files']:,} files extracted (archive {ex['archive_files']:,}), "
                f"{fp.get('audio', 0):,} of {fp['videos']:,} videos have audio, {fp.get('error', 0)} unreadable, tar deleted {deleted}; details in DATA_LOG.md.")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # report instead of dying silently
        status_line(f"LAV-DF job FAILED: {type(e).__name__}: {str(e)[:200]}; rerun scripts/data/lavdf_job.py to resume.")
        raise
