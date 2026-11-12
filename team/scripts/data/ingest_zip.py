"""Ingest a zip archive that the user downloaded by hand: verify, extract, check counts, record the hash.

Usage: python scripts/data/ingest_zip.py <archive.zip> <dest folder> <manifest name> [folder=count ...] [file=lines ...]
Example: python scripts/data/ingest_zip.py C:/.../Celeb-DF-v2.zip data/celebdf_v2 celebdf_v2_zip \
             Celeb-real=590 YouTube-real=300 Celeb-synthesis=5639 List_of_testing_videos.txt=518

Each `name=count` argument is checked against the number of files below the top-level folder `name`
(or, for a name ending in .txt, against the number of non-empty lines of that file after extraction).
The archive is left where it is; the user decides whether to delete it.
"""
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_open as fo  # noqa: E402


def main():
    archive, dest, name, *checks = sys.argv[1:]
    archive, dest = Path(archive), Path(dest)
    fo.say(f"hashing {archive.name} ({archive.stat().st_size} bytes)")
    sha = fo.sha256_file(archive)
    with zipfile.ZipFile(archive) as z:
        bad = z.testzip()
        if bad:
            raise SystemExit(f"corrupt member {bad}")
        fo.say("all member CRCs verified")
        unsafe = [i.filename for i in z.infolist() if i.filename.startswith("/") or ".." in i.filename.split("/")]
        if unsafe:
            raise SystemExit(f"unsafe member names: {unsafe[:3]}")
        dest.mkdir(parents=True, exist_ok=True)
        z.extractall(dest)
        counts = {}
        for i in z.infolist():
            if not i.is_dir():
                top = i.filename.split("/")[0]
                counts[top] = counts.get(top, 0) + 1
    fo.say(f"extracted to {dest}; files per top-level entry: {counts}")
    for c in checks:
        key, want = c.rsplit("=", 1)
        want = int(want)
        if key.endswith(".txt"):
            got = sum(1 for l in open(dest / key, encoding="utf-8") if l.strip())
        else:
            got = counts.get(key, 0)
        fo.say(f"check {key}: found {got}, documented {want}: {'OK' if got == want else 'MISMATCH'}")
        if got != want:
            raise SystemExit(f"{key}: {got} differs from documented {want}")
    fo.record(name, "user download (Google Drive link from the terms page)", archive,
              archive.stat().st_size, sha, "none published, recorded only", {"files_per_top_level": counts})


if __name__ == "__main__":
    main()
