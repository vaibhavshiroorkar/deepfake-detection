"""Threaded download of FaceForensics++ videos from the dataset's own server (after approval, D-020).

The authors' script downloads one file at a time, and each connection to the server is slow, so this
tool fetches many files at once from the same URLs with the same file lists (misc/filelist.json and
misc/deepfake_detection_filenames.json). Files are written to a temp name and renamed, existing files
are skipped, errors are retried. Only the datasets chosen in D-021 are known to it.

Usage: python scripts/data/ffpp_parallel.py <threads> <dataset> [<dataset> ...]
Datasets: original Deepfakes Face2Face FaceSwap NeuralTextures DeepFakeDetection_original DeepFakeDetection
"""
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = "http://kaldir.vc.in.tum.de/faceforensics/v3/"  # server EU2, the only available one (approval email)
PATHS = {
    "original": ("data/ffpp", "original_sequences/youtube"),
    "Deepfakes": ("data/ffpp", "manipulated_sequences/Deepfakes"),
    "Face2Face": ("data/ffpp", "manipulated_sequences/Face2Face"),
    "FaceSwap": ("data/ffpp", "manipulated_sequences/FaceSwap"),
    "NeuralTextures": ("data/ffpp", "manipulated_sequences/NeuralTextures"),
    "DeepFakeDetection_original": ("data/dfd", "original_sequences/actors"),
    "DeepFakeDetection": ("data/dfd", "manipulated_sequences/DeepFakeDetection"),
}


def get_json(rel):
    with urllib.request.urlopen(BASE + rel, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def file_list(dataset):
    if dataset.startswith("DeepFakeDetection"):
        d = get_json("misc/deepfake_detection_filenames.json")
        return d["actors"] if dataset == "DeepFakeDetection_original" else d["DeepFakesDetection"]
    pairs = get_json("misc/filelist.json")
    if dataset == "original":
        return [x for p in pairs for x in p]
    out = []
    for p in pairs:
        out.append("_".join(p))
        out.append("_".join(p[::-1]))
    return out


def fetch(job):
    url, dest = job
    dest = Path(dest)
    if dest.exists():
        return True
    tmp = dest.with_name("part_" + dest.name)
    for attempt in range(8):
        try:
            with urllib.request.urlopen(url, timeout=90) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            tmp.replace(dest)
            return True
        except Exception:
            time.sleep(3 + 2 * attempt)
    if tmp.exists():
        tmp.unlink()
    return False


def main():
    threads = int(sys.argv[1])
    for dataset in sys.argv[2:]:
        out, sub = PATHS[dataset]
        folder = ROOT / out / sub / "c23" / "videos"
        folder.mkdir(parents=True, exist_ok=True)
        for p in folder.glob("tmp*"):
            p.unlink(missing_ok=True)
        for p in folder.glob("part_*"):
            p.unlink(missing_ok=True)
        names = [n + ".mp4" for n in file_list(dataset)]
        todo = [(f"{BASE}{sub}/c23/videos/{n}", folder / n) for n in names if not (folder / n).exists()]
        print(f"[{time.strftime('%FT%T')}] {dataset}: {len(names)} files, {len(todo)} to fetch with {threads} threads", flush=True)
        t0 = time.time()
        done = failed = 0
        with ThreadPoolExecutor(threads) as ex:
            for ok in ex.map(fetch, todo):
                done += ok
                failed += (not ok)
                if (done + failed) % 50 == 0:
                    print(f"[{time.strftime('%FT%T')}] {dataset}: {done + failed}/{len(todo)} "
                          f"({failed} failed), {time.time() - t0:.0f} s", flush=True)
        have = sum(1 for _ in folder.glob("*.mp4"))
        print(f"[{time.strftime('%FT%T')}] {dataset}: finished, {have}/{len(names)} on disk, {failed} failed", flush=True)


if __name__ == "__main__":
    main()
