"""Writes submission/MANIFEST.txt: the SHA-256 and size of every file in submission/ (MANIFEST.txt itself excluded).
Run from the repository root: python scripts/eval/make_submission_manifest.py
"""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "submission"
rows = []
for p in sorted(ROOT.rglob("*")):
    if p.is_file() and p.name != "MANIFEST.txt":
        rows.append((p.relative_to(ROOT).as_posix(), hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_size))
with open(ROOT / "MANIFEST.txt", "w", newline="\n") as m:
    m.write("SHA-256 manifest of submission/ (generated 2026-10-29; MANIFEST.txt itself excluded)\n")
    for n, h, s in rows:
        m.write("%s  %s  %d\n" % (h, n, s))
print(len(rows), "files,", round(sum(r[2] for r in rows) / 1e6, 2), "MB")
