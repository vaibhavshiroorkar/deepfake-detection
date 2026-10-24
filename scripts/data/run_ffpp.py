"""Run the FaceForensics++ download script of the dataset authors with a network timeout and retries.

The authors' script has no timeout, so a hung connection stalls it, and a dropped connection ends it.
This wrapper sets a socket timeout, answers the terms-of-use prompt with Enter (the user accepted the
terms in the request form), runs the script for one dataset, and repeats until the expected number of
videos is on disk or the attempts are used up. The script skips files that already exist.

Usage: python scripts/data/run_ffpp.py <dataset> [attempts]
Datasets: original Deepfakes Face2Face FaceSwap NeuralTextures DeepFakeDetection_original DeepFakeDetection
Never pass "all" (it would also pull DFD and FaceShifter).
"""
import io
import os
import runpy
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "download-FaceForensics.py"
LAYOUT = {  # dataset -> (output folder, sub path, expected video count)
    "original": ("data/ffpp", "original_sequences/youtube", 1000),
    "Deepfakes": ("data/ffpp", "manipulated_sequences/Deepfakes", 1000),
    "Face2Face": ("data/ffpp", "manipulated_sequences/Face2Face", 1000),
    "FaceSwap": ("data/ffpp", "manipulated_sequences/FaceSwap", 1000),
    "NeuralTextures": ("data/ffpp", "manipulated_sequences/NeuralTextures", 1000),
    "DeepFakeDetection_original": ("data/dfd", "original_sequences/actors", 363),
    "DeepFakeDetection": ("data/dfd", "manipulated_sequences/DeepFakeDetection", 3068),
}


def count(out, sub):
    d = ROOT / out / sub / "c23" / "videos"
    return sum(1 for p in d.glob("*.mp4")) if d.exists() else 0


def clean_tmp(out, sub):
    d = ROOT / out / sub / "c23" / "videos"
    if d.exists():
        for p in d.glob("tmp*"):
            try:
                p.unlink()
            except OSError:
                pass


def main():
    dataset = sys.argv[1]
    attempts = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    out, sub, want = LAYOUT[dataset]
    os.chdir(ROOT)
    for k in range(1, attempts + 1):
        have = count(out, sub)
        print(f"[{time.strftime('%FT%T')}] {dataset}: attempt {k}, {have}/{want} videos on disk", flush=True)
        if have >= want:
            print(f"[{time.strftime('%FT%T')}] {dataset}: complete", flush=True)
            return 0
        socket.setdefaulttimeout(120)
        sys.stdin = io.StringIO("\n")
        sys.argv = [str(SCRIPT), out, "-d", dataset, "-c", "c23", "-t", "videos", "--server", "EU2"]
        try:
            runpy.run_path(str(SCRIPT), run_name="__main__")
        except SystemExit:
            pass
        except Exception as e:  # network errors: clean up partial temp files and try again
            print(f"[{time.strftime('%FT%T')}] {dataset}: error {type(e).__name__}: {e}", flush=True)
        clean_tmp(out, sub)
        time.sleep(5)
    print(f"[{time.strftime('%FT%T')}] {dataset}: gave up with {count(out, sub)}/{want}", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
