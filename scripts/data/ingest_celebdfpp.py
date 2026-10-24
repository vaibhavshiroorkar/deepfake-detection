"""Celeb-DF++: verify the downloaded zip and extract only what D-021 needs.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/data/ingest_celebdfpp.py data/_incoming/Celeb-DF++.zip data/celebdf_pp

Steps: SHA-256 of the zip, a CRC test of every member, the layout counts of the whole archive, then extraction of
  * List_of_testing_videos.txt (kept whole, 5,418 lines),
  * the listed real videos (Celeb-real and YouTube-real, 178),
  * the listed fakes of the 21 NEW generators (4,900); the 340 listed fakes of the folder Celeb-DF-v2 are NOT extracted
    because the same videos are in the Celeb-DF v2 archive already (D-021).
The root folder `Celeb-DF-v3/` of the archive is dropped. The zip is left where it is (kept until the freeze).
Everything is written to data/_logs/celebdfpp_ingest.json. Members are read with the zipfile CRC check, so a bad member raises.
"""
from __future__ import annotations

import collections
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

ROOT = "Celeb-DF-v3/"
OLD_FOLDER = "Celeb-synthesis/FaceSwap/Celeb-DF-v2/"
LOG = Path("data/_logs/celebdfpp_ingest.json")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def main(archive: str, dest: str) -> int:
    archive, dest = Path(archive), Path(dest)
    res = {"archive": archive.name, "bytes": archive.stat().st_size, "started": time.strftime("%Y-%m-%d %H:%M")}
    LOG.parent.mkdir(parents=True, exist_ok=True)
    res["sha256"] = sha256(archive)
    with zipfile.ZipFile(archive) as z:
        bad = z.testzip()          # reads and CRC-checks every member
        res["crc_bad_member"] = bad
        if bad:
            LOG.write_text(json.dumps(res, indent=1))
            raise SystemExit(f"corrupt member {bad}")
        infos = [i for i in z.infolist() if not i.is_dir()]
        unsafe = [i.filename for i in infos if i.filename.startswith("/") or ".." in i.filename.split("/")]
        if unsafe:
            raise SystemExit(f"unsafe member names {unsafe[:3]}")
        layout = collections.Counter()
        for i in infos:
            p = i.filename[len(ROOT):].split("/")
            layout["/".join(p[:3]) if p[0] == "Celeb-synthesis" and len(p) > 3 else p[0]] += 1
        res["archive_files"] = len(infos)
        res["layout_counts"] = dict(sorted(layout.items()))
        lst = z.read(ROOT + "List_of_testing_videos.txt").decode("utf-8")
        lines = [l for l in lst.splitlines() if l.strip()]
        listed = [l.split(" ", 1)[1].strip() for l in lines]
        want_real = [p for p in listed if p.startswith(("Celeb-real/", "YouTube-real/"))]
        want_fake = [p for p in listed if p.startswith("Celeb-synthesis/") and not p.startswith(OLD_FOLDER)]
        skipped_old = [p for p in listed if p.startswith(OLD_FOLDER)]
        names = {i.filename: i for i in infos}
        missing = [p for p in want_real + want_fake if ROOT + p not in names]
        res.update(list_lines=len(lines), listed_real=len(want_real), listed_new_fakes=len(want_fake), skipped_celebdf_v2_fakes=len(skipped_old), listed_missing_in_zip=len(missing))
        if missing:
            raise SystemExit(f"{len(missing)} listed videos are not in the zip, first {missing[:3]}")
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "List_of_testing_videos.txt").write_text(lst, encoding="utf-8", newline="")
        n = size = 0
        per = collections.Counter()
        for p in want_real + want_fake:
            info = names[ROOT + p]
            out = dest / p
            out.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(out, "wb") as dst:   # the CRC is verified when the member is read to its end
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
            if out.stat().st_size != info.file_size:
                raise SystemExit(f"size mismatch for {p}")
            n += 1
            size += info.file_size
            per["/".join(p.split("/")[:3]) if p.startswith("Celeb-synthesis") else p.split("/")[0]] += 1
        res.update(extracted_files=n, extracted_bytes=size, extracted_per_folder=dict(sorted(per.items())), finished=time.strftime("%Y-%m-%d %H:%M"))
    LOG.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("layout_counts", "extracted_per_folder")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
