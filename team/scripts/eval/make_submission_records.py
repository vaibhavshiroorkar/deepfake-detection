"""Copies the key records into submission/records/ and removes per-file rows of the datasets whose terms forbid derived data.

Celeb-DF, Celeb-DF++ and FakeAVCeleb (docs/DATA_LOG.md): no frame, crop or per-video score may be distributed. docs/EXPERIMENTS.md holds a few
tables that name single files of those sets with their scores; those rows are replaced by a marker line in the COPY only. The records in docs/ are
not changed. The copies are written with LF line endings (the form git stores), whatever the checkout uses (D-093).
Run from the repository root: python scripts/eval/make_submission_records.py
"""
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "submission" / "records"
OUT.mkdir(parents=True, exist_ok=True)
FILE_ROW = re.compile(r"(celeb|fakeavceleb|youtube-real|id\d+_\d+)[^\n]*\.(mp4|avi|wav|png|jpg|jpeg|flac)", re.IGNORECASE)
MARK = "| (row removed from this copy: it names a single file of a data set whose terms forbid derived data) |"
for name in ("DECISIONS.md", "EXPERIMENTS.md", "STATUS.md", "PROTOCOL.md"):
    src = ROOT / "docs" / name
    text = src.read_bytes().decode("utf-8").replace("\r\n", "\n")
    lines = text.split("\n")
    out, removed = [], 0
    for ln in lines:
        if ln.startswith("|") and FILE_ROW.search(ln):
            out.append(MARK)
            removed += 1
        else:
            out.append(ln)
    (OUT / name).write_bytes("\n".join(out).encode("utf-8"))
    print(name, "rows removed:", removed)
