"""Writes submission/MANIFEST.txt: the SHA-256 and size of every file in submission/ (MANIFEST.txt itself excluded).
Text files (TEXT_SUFFIXES) are hashed and sized in their LF form, the bytes git stores, so the manifest does not depend on the
line endings of the checkout (core.autocrlf on Windows, D-093); every other file is hashed as it is on disk.
Run from the repository root: python scripts/eval/make_submission_manifest.py
"""
import datetime
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "submission"
TEXT_SUFFIXES = {".md", ".tex", ".bib", ".bst", ".cls", ".txt"}
rows = []
for p in sorted(ROOT.rglob("*")):
    if p.is_file() and p.name != "MANIFEST.txt":
        data = p.read_bytes()
        if p.suffix.lower() in TEXT_SUFFIXES:
            data = data.replace(b"\r\n", b"\n")
        rows.append((p.relative_to(ROOT).as_posix(), hashlib.sha256(data).hexdigest(), len(data)))
with open(ROOT / "MANIFEST.txt", "w", newline="\n") as m:
    m.write("SHA-256 manifest of submission/ (generated %s; MANIFEST.txt itself excluded; text files %s hashed and sized with LF line endings,"
            " as stored by git; other files as on disk)\n" % (datetime.date.today().isoformat(), " ".join(sorted(TEXT_SUFFIXES))))
    for n, h, s in rows:
        m.write("%s  %s  %d\n" % (h, n, s))
print(len(rows), "files,", round(sum(r[2] for r in rows) / 1e6, 2), "MB")
