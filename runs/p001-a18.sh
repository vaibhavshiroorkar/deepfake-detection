#!/usr/bin/env bash
# P-001 amendment A18 (exploratory): H1 again with a fixed video bitrate (160 kb/s constant) instead of
# constant quality. All of FakeAVCeleb, written beside v4 in data/neutral-cbr, then the unchanged audit.
# CPU only (ffmpeg and ffprobe); it can run beside the GPU checks of A16 and A19.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
stamp() { echo "[$(date '+%a %H:%M:%S')] $*"; }
stamp "A18 neutralize all of FakeAVCeleb at 160k"
$PY scripts/neutralize_dataset.py --manifest data/manifests/fakeavceleb.csv --dataset FakeAVCeleb \
    --output-root data/neutral-cbr --manifest-out data/manifests/fakeavceleb-neutral-cbr.csv \
    --workers 8 --video-bitrate 160k | grep -A16 rows_failed
stamp "A18 audit"
$PY scripts/shortcut_audit.py --manifest data/manifests/fakeavceleb-neutral-cbr.csv --dataset FakeAVCeleb-neutral-cbr \
    --output results/shortcut-audit/fakeavceleb-neutral-cbr --workers 8 | grep -v '^[0-9]*/[0-9]*$'
stamp "A18 done"
