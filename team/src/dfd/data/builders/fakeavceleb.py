"""FakeAVCeleb v1.2 builder (layout verified on the archive on 2026-10-26; D-070).

Layout (archive FakeAVCeleb_v1.2.zip, extracted by the user-approved step of 2026-10-26 into data/fakeavceleb/):
  FakeAVCeleb_v1.2/<Category>/<Race>/<gender>/<idNNNNN>/<file>.mp4     21,544 files
  FakeAVCeleb_v1.2/meta_data.csv       21,566 rows (22 paths appear twice); columns source, target1, target2, method, category (A to D),
                                       type, race, gender, filename, path ("FakeAVCeleb/<Category>/<Race>/<gender>/<id>")
  Categories (type column): RealVideo-RealAudio (RVRA, 500), RealVideo-FakeAudio (RVFA, 500, RTVC voice cloning),
  FakeVideo-RealAudio (FVRA, 9,709), FakeVideo-FakeAudio (FVFA, 10,857). Methods: real, rtvc, wav2lip, fsgan, faceswap and combinations.

Rules.
  * Duplicated metadata paths (22, all FakeVideo-FakeAudio) differ only in the ``method`` column (faceswap-wav2lip against wav2lip): one record, method written
    ``ambiguous:<a>|<b>``, excluded from any per-method breakdown.
  * A fake-video clip with no identity other than its source in the name or target columns (534) reuses the source's own face (Wav2Lip on the same person);
    its identity set is the source alone and notes carry ``other_target=false``.
  * One record per archive member. A metadata path that occurs twice is collapsed to one record; the report lists every duplicate and
    whether its rows agree.
  * Identities are the VoxCeleb2 speaker ids ``idNNNNN``. The source identity is the ``source`` column (equal to the folder id). The target
    identities are the ids found in the file name and in the ``target1`` and ``target2`` columns; those columns also hold other tokens
    (numbers, ``fake.mp4``, ``faceswap``, YouTube-style ids) which are ignored.
  * group_id = the source identity (group_basis identity). The full identity set of a clip goes to ``notes`` as ``ids=src|t1|t2``.
    Groups are NOT closed under the target relation; the identity-separated split of D-070 (``dfd.data.fakeavceleb_split``) excludes clips whose
    identities fall on both sides.
  * label = 1 when the video or the audio is fake (RVFA counts as fake); the per-modality label is kept in the notes as ``category=``.
  * The source family is ``fakeavceleb``; VoxCeleb2 is also the source of the LAV-DF real videos (a possible shared family, not verifiable by id).
"""
from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

from ..manifest import Record

KEY = "fakeavceleb"
FAMILY = "fakeavceleb"
ROOT = "FakeAVCeleb_v1.2"
ID_RE = re.compile(r"id\d{5}")
FULL_ID = re.compile(r"^id\d{5}$")
CATEGORY_CODE = {"RealVideo-RealAudio": "RVRA", "RealVideo-FakeAudio": "RVFA", "FakeVideo-RealAudio": "FVRA", "FakeVideo-FakeAudio": "FVFA"}
EXPECTED = {"RVRA": 500, "RVFA": 500, "FVRA": 9709, "FVFA": 10857}
EXPECTED_MEMBERS = 21544
EXPECTED_ROWS = 21566


def target_ids(source: str, filename: str, target1: str, target2: str) -> list[str]:
    """Target identities of a clip: ids in the file name and in the target columns, in order, without repeats, without the source."""
    out: list[str] = []
    for tok in ID_RE.findall(filename) + [t for t in (target1, target2) if FULL_ID.match(t)]:
        if tok != source and tok not in out:
            out.append(tok)
    return out


def member_path(meta_path: str, filename: str) -> str:
    """Archive-relative path (below FakeAVCeleb_v1.2/) of a metadata row."""
    base = meta_path.strip().strip("/")
    if base.startswith("FakeAVCeleb/"):
        base = base[len("FakeAVCeleb/"):]
    return f"{base}/{filename.strip()}"


def read_metadata(csv_path: Path) -> list[dict]:
    with open(csv_path, encoding="utf-8", errors="replace", newline="") as fh:
        rows = list(csv.reader(fh))
    hdr = [h.strip() for h in rows[0]]
    want = ["source", "target1", "target2", "method", "category", "type", "race", "gender"]
    if hdr[:8] != want:
        raise ValueError(f"unexpected meta_data.csv header {hdr}")
    out = []
    for r in rows[1:]:
        if not any(c.strip() for c in r):
            continue
        out.append({"source": r[0].strip(), "target1": r[1].strip(), "target2": r[2].strip(), "method": r[3].strip(),
                    "letter": r[4].strip(), "type": r[5].strip(), "race": r[6].strip(), "gender": r[7].strip(),
                    "filename": r[8].strip(), "path": r[9].strip()})
    return out


def build(data_root: Path) -> tuple[list[Record], dict]:
    base = data_root / KEY / ROOT
    meta = base / "meta_data.csv"
    if not meta.is_file():
        raise FileNotFoundError(f"{meta} not found: extract FakeAVCeleb_v1.2.zip into {data_root / KEY} first (D-070)")
    rows = read_metadata(meta)
    on_disk = {p.relative_to(base).as_posix() for p in base.rglob("*.mp4")}
    by_path: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_path[member_path(r["path"], r["filename"])].append(r)
    duplicates = []
    for p, rs in by_path.items():
        if len(rs) > 1:
            same = all(x == rs[0] for x in rs)
            duplicates.append({"path": p, "rows": len(rs), "rows_identical": same})
    missing_file = sorted(p for p in by_path if p not in on_disk)
    no_metadata = sorted(on_disk - set(by_path))
    records, no_target, cats, methods, bad = [], [], Counter(), Counter(), []
    for p in sorted(by_path):
        if p in missing_file:
            continue
        r = dict(by_path[p][0])
        if len(by_path[p]) > 1:
            ms = sorted({x["method"] for x in by_path[p]})
            if len(ms) > 1:       # the 22 duplicates differ only in the method column: keep the clip, mark the method as ambiguous
                r["method"] = "ambiguous:" + "|".join(ms)
        code = CATEGORY_CODE.get(r["type"])
        if code is None:
            bad.append({"path": p, "type": r["type"]})
            continue
        src = r["source"]
        tgt = target_ids(src, r["filename"], r["target1"], r["target2"])
        fake_clip = code != "RVRA"
        if code in ("FVRA", "FVFA") and not tgt:
            no_target.append(p)
        cell = f"{r['race']}/{r['gender']}"
        label = 1 if fake_clip else 0
        method = r["method"] if fake_clip else "real"
        cats[code] += 1
        methods[method] += 1
        notes = f"category={code};ids={'|'.join([src] + tgt)};cell={cell};other_target={'true' if tgt else 'false'}"
        local = p[:-4] if p.endswith(".mp4") else p
        records.append(Record(
            sample_id=f"{KEY}:{local}", dataset=KEY, family=FAMILY, path=f"{KEY}/{ROOT}/{p}", modality="video", label=label, method=method,
            group_id=f"{KEY}:{src}", group_basis="identity", identity_known=True, official_split="", notes=notes))
    report = {
        "members_on_disk": len(on_disk), "expected_members": EXPECTED_MEMBERS, "metadata_rows": len(rows), "expected_rows": EXPECTED_ROWS,
        "unique_metadata_paths": len(by_path), "duplicate_paths": len(duplicates), "duplicate_detail": duplicates,
        "metadata_rows_without_file": missing_file, "files_without_metadata": no_metadata, "unknown_type": bad,
        "records": len(records), "categories": dict(cats), "expected_categories": EXPECTED, "methods": dict(methods),
        "fake_video_clips_without_other_target": len(no_target), "fake_video_clips_without_other_target_examples": no_target[:10],
        "sources": len({r.group_id for r in records}),
    }
    return records, report
