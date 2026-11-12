"""Celeb-DF v2 builder and Celeb-DF++ builder.

Celeb-DF v2 layout (DATA_LOG.md and the archive listing seen by the main agent):
  data/celebdf_v2/Celeb-real/id<N>_<k>.mp4              590 videos, real, celebrity id<N>        e.g. id0_0000.mp4
  data/celebdf_v2/YouTube-real/<name>.mp4               300 videos, real, YouTube                (names seen: 00000.mp4 style)
  data/celebdf_v2/Celeb-synthesis/id<T>_id<S>_<k>.mp4   5,639 videos, fake                       e.g. id0_id16_0000.mp4
  data/celebdf_v2/List_of_testing_videos.txt            518 lines (the official test list)

Rules.
  * Celeb-real: group = the identity in the name (identity_known true).
  * Celeb-synthesis: a fake joins two identities. Assumption: the first id (T) is the target video whose
    identity labels the group, the second (S) is the face source. group_id uses T; both ids are written to
    ``notes`` as ``ids=T|S`` so that identity overlap checks can use both. A group is therefore NOT a
    connected component over the identity graph (measured on the real files: 3 components of 38, 10 and 10
    identities, so a component group would be very coarse for bootstrap resampling); this is a choice,
    listed in docs/_agents/S2B.md.
  * YouTube-real: the file names (00000.mp4 style, seen on disk) carry no identity, so the group is the video
    itself (group_basis source_video, identity_known false).
  * Official split: the test list is parsed tolerantly (any whitespace-separated token ending in .mp4 is a
    path; the leading 0/1 token is ignored; in the seen file YouTube-real lines start with 1). Listed videos get
    official_split ``test``; others get an empty value.
  * Celeb-DF v2 and Celeb-DF++ are ONE source family (``celebdf``), D-012 and DATA_LOG.md.

Celeb-DF++ (``celebdf_pp``): see build_pp (layout verified on the archive on 2026-10-25).
"""

from __future__ import annotations

import re
from pathlib import Path

from ..manifest import Record, rel_posix
from . import LayoutNotVerified

KEY = "celebdf_v2"
FAMILY = "celebdf"
EXPECTED = {"Celeb-real": 590, "YouTube-real": 300, "Celeb-synthesis": 5639}
EXPECTED_TEST_LIST_LINES = 518

_REAL_RE = re.compile(r"^(id\d+)_(\d+)\.mp4$")
_FAKE_RE = re.compile(r"^(id\d+)_(id\d+)_(\d+)\.mp4$")


def parse_celeb_name(folder: str, filename: str) -> dict:
    """Parse a Celeb-DF file name. Returns dict(label, target, source, identity_known). Raises ValueError."""
    if folder == "Celeb-real":
        m = _REAL_RE.match(filename)
        if not m:
            raise ValueError(f"Celeb-real name not id<N>_<k>.mp4: {filename}")
        return {"label": 0, "target": m.group(1), "source": None, "identity_known": True}
    if folder == "Celeb-synthesis":
        m = _FAKE_RE.match(filename)
        if not m:
            raise ValueError(f"Celeb-synthesis name not id<T>_id<S>_<k>.mp4: {filename}")
        return {"label": 1, "target": m.group(1), "source": m.group(2), "identity_known": True}
    if folder == "YouTube-real":
        if not filename.lower().endswith(".mp4"):
            raise ValueError(f"YouTube-real name not .mp4: {filename}")
        return {"label": 0, "target": None, "source": None, "identity_known": False}
    raise ValueError(f"unknown Celeb-DF folder {folder}")


def parse_test_list(text: str) -> set[str]:
    out = set()
    for line in text.splitlines():
        for tok in line.split():
            if tok.lower().endswith(".mp4"):
                out.add(tok.replace("\\", "/"))
    return out


def build(data_root: Path) -> tuple[list[Record], dict]:
    base = data_root / "celebdf_v2"
    tl = base / "List_of_testing_videos.txt"
    test_lines = tl.read_text(encoding="utf-8").splitlines() if tl.exists() else []
    test_set = parse_test_list("\n".join(test_lines))
    records: list[Record] = []
    found: dict[str, int] = {}
    bad: list[str] = []
    for folder in EXPECTED:
        d = base / folder
        n = 0
        if d.is_dir():
            for p in sorted(d.iterdir()):
                if p.suffix.lower() != ".mp4" or not p.is_file():
                    bad.append(rel_posix(p, data_root))
                    continue
                try:
                    info = parse_celeb_name(folder, p.name)
                except ValueError:
                    bad.append(rel_posix(p, data_root))
                    continue
                n += 1
                rel_in = f"{folder}/{p.name}"
                if info["target"] is not None:
                    group, basis = f"{KEY}:{info['target']}", "identity"
                    notes = "ids=" + "|".join(x for x in (info["target"], info["source"]) if x)
                else:
                    group, basis, notes = f"{KEY}:youtube:{p.stem}", "source_video", ""
                records.append(
                    Record(
                        sample_id=f"{KEY}:{rel_in}",
                        dataset=KEY,
                        family=FAMILY,
                        path=rel_posix(p, data_root),
                        modality="video",
                        label=info["label"],
                        method="real" if info["label"] == 0 else "celebdf_synthesis",
                        group_id=group,
                        group_basis=basis,
                        identity_known=info["identity_known"],
                        official_split="test" if rel_in in test_set else "",
                        notes=notes,
                    )
                )
        found[folder] = n
    listed_missing = sorted(t for t in test_set if not (base / t).is_file())
    report = {
        "dataset": KEY,
        "found": found,
        "expected": EXPECTED,
        "rows": len(records),
        "test_list_lines": len(test_lines),
        "test_list_lines_expected": EXPECTED_TEST_LIST_LINES,
        "test_list_entries_missing_on_disk": len(listed_missing),
        "unparsed_or_ignored_files": bad,
        "complete": found == EXPECTED and len(test_lines) == EXPECTED_TEST_LIST_LINES and not listed_missing,
    }
    return records, report


KEY_PP = "celebdf_pp"
EXPECTED_PP_REAL = {"Celeb-real": 108, "YouTube-real": 70}  # the listed real videos of the official test list
NEW_FOLDERS = {  # type -> {method: listed videos}: the 21 NEW generators; the Celeb-DF-v2 folder is not used (D-021)
    "FaceReenact": {m: 200 for m in ("DaGAN", "FSRT", "HyperReenact", "LIA", "LivePortrait", "MCNET", "TPSMM")},
    "FaceSwap": {m: 200 for m in ("BlendFace", "GHOST", "HifiFace", "InSwapper", "MobileFaceSwap", "SimSwap", "UniFace")},
    "TalkingFace": {m: 300 for m in ("AniTalker", "EDTalk", "EchoMimic", "FLOAT", "IP_LAP", "Real3DPortrait", "SadTalker")},
}
EXPECTED_PP_TEST_LIST_LINES = 5418
_TALK_RE = re.compile(r"^(id\d+)_(\d+)_test_(id\d+)_(.+)\.mp4$")


def parse_pp_fake_name(ftype: str, filename: str) -> dict:
    """Celeb-DF++ fake names. FaceSwap and FaceReenact: id<T>_id<S>_<k>.mp4 (target, face source). TalkingFace:
    id<T>_<k>_test_id<V>_<video>.mp4 (target celebrity, VoxCeleb speaker and video of the driving audio)."""
    if ftype in ("FaceSwap", "FaceReenact"):
        m = _FAKE_RE.match(filename)
        if not m:
            raise ValueError(f"{ftype} name not id<T>_id<S>_<k>.mp4: {filename}")
        return {"target": m.group(1), "source": m.group(2)}
    m = _TALK_RE.match(filename)
    if not m:
        raise ValueError(f"TalkingFace name not id<T>_<k>_test_id<V>_<video>.mp4: {filename}")
    return {"target": m.group(1), "source": m.group(3)}


def build_pp(data_root: Path) -> tuple[list[Record], dict]:
    """Celeb-DF++ test-list videos: data/celebdf_pp/{Celeb-real,YouTube-real,Celeb-synthesis/<type>/<method>}/, one family
    (``celebdf``) with Celeb-DF v2 (D-012). Group = the target identity of the name (as in Celeb-DF v2, D-026); both identities
    are written to ``notes`` as ``ids=T|S`` so that identity checks can use them. Method = the generator folder. Every row is
    in the official test list, so official_split is ``test``."""
    base = data_root / "celebdf_pp"
    tl = base / "List_of_testing_videos.txt"
    test_lines = tl.read_text(encoding="utf-8").splitlines() if tl.exists() else []
    test_set = parse_test_list("\n".join(test_lines))
    records: list[Record] = []
    found: dict[str, int] = {}
    bad: list[str] = []

    def add(p: Path, rel_in: str, label: int, method: str, group: str, basis: str, known: bool, notes: str):
        records.append(Record(sample_id=f"{KEY_PP}:{rel_in}", dataset=KEY_PP, family=FAMILY, path=rel_posix(p, data_root), modality="video",
                              label=label, method=method, group_id=group, group_basis=basis, identity_known=known,
                              official_split="test" if rel_in in test_set else "", notes=notes))

    for folder in EXPECTED_PP_REAL:
        d, n = base / folder, 0
        if d.is_dir():
            for p in sorted(d.iterdir()):
                try:
                    info = parse_celeb_name(folder, p.name)
                except ValueError:
                    bad.append(rel_posix(p, data_root))
                    continue
                n += 1
                rel_in = f"{folder}/{p.name}"
                if info["target"] is not None:
                    add(p, rel_in, 0, "real", f"{KEY_PP}:{info['target']}", "identity", True, f"ids={info['target']}")
                else:
                    add(p, rel_in, 0, "real", f"{KEY_PP}:youtube:{p.stem}", "source_video", False, "")
        found[folder] = n
    for ftype, methods in NEW_FOLDERS.items():
        for method in methods:
            d, n = base / "Celeb-synthesis" / ftype / method, 0
            if d.is_dir():
                for p in sorted(d.iterdir()):
                    try:
                        info = parse_pp_fake_name(ftype, p.name)
                    except ValueError:
                        bad.append(rel_posix(p, data_root))
                        continue
                    n += 1
                    add(p, f"Celeb-synthesis/{ftype}/{method}/{p.name}", 1, method, f"{KEY_PP}:{info['target']}", "identity", True,
                        f"ids={info['target']}|{info['source']}")
            found[f"{ftype}/{method}"] = n
    expected = dict(EXPECTED_PP_REAL)
    expected.update({f"{t}/{m}": c for t, ms in NEW_FOLDERS.items() for m, c in ms.items()})
    listed_new = {t for t in test_set if t.startswith(("Celeb-real/", "YouTube-real/")) or (t.startswith("Celeb-synthesis/") and "/Celeb-DF-v2/" not in t)}
    listed_missing = sorted(t for t in listed_new if not (base / t).is_file())
    report = {"dataset": KEY_PP, "found": found, "expected": expected, "rows": len(records), "test_list_lines": len(test_lines),
              "test_list_lines_expected": EXPECTED_PP_TEST_LIST_LINES, "test_list_entries_missing_on_disk": len(listed_missing),
              "unparsed_or_ignored_files": bad, "not_extracted_by_design": "340 listed fakes of the folder Celeb-DF-v2 (already in Celeb-DF v2)",
              "complete": found == expected and len(test_lines) == EXPECTED_PP_TEST_LIST_LINES and not listed_missing and not bad}
    return records, report
