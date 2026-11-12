"""Google DeepFakeDetection (DFD) c23 manifest builder (test_x only).

Layout (verified on the real folders, 2026-10-24):
  data/dfd/original_sequences/actors/c23/videos/<actor>__<scene>.mp4                                   363 files
  data/dfd/manipulated_sequences/DeepFakeDetection/c23/videos/<target>_<source>__<sequence>__<experiment id>.mp4   3,068 files
Actors are two-digit numbers (01 to 28 on the real files).

Group rule. The actor identity is in the file name. A real video belongs to its actor. A fake joins two
actors; as for Celeb-DF (D-026) the group is the FIRST actor (the target, assumed to be the video that was
kept; not verified), and the source actor is written to ``notes`` as ``source=<actor>``. A connected component over the actor graph is the strict alternative; measured on the real files it gives 3 components (18, 5 and 5 actors), too coarse for a bootstrap cluster. group_basis identity, identity_known true.

method = ``DeepFakeDetection`` (the single manipulation method of this folder); the experiment id is kept
in notes. Family ``dfd``. No official split. DFD is a test_x dataset; it is never in train.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from ..manifest import Record, rel_posix
from .ffpp import components

KEY = "dfd"
FAMILY = "dfd"
EXPECTED_ORIGINALS = 363
EXPECTED_MANIPULATED = 3068

_ORIG_RE = re.compile(r"^(?P<actor>\d{2})__(?P<scene>[^_].*?)\.mp4$")
_MANIP_RE = re.compile(r"^(?P<target>\d{2})_(?P<source>\d{2})__(?P<seq>.+?)__(?P<exp>[A-Za-z0-9]+)\.mp4$")


def parse_dfd_name(filename: str) -> dict:
    """original ``<actor>__<scene>.mp4``; manipulated ``<target>_<source>__<sequence>__<experiment id>.mp4``."""
    m = _MANIP_RE.match(filename)
    if m:
        return {"label": 1, "target": m["target"], "source": m["source"], "scene_or_sequence": m["seq"],
                "experiment": m["exp"]}
    m = _ORIG_RE.match(filename)
    if m:
        return {"label": 0, "target": m["actor"], "source": None, "scene_or_sequence": m["scene"],
                "experiment": None}
    raise ValueError(f"not a DFD file name: {filename}")


def build(data_root: Path) -> tuple[list[Record], dict]:
    base = data_root / "dfd"
    dirs = [
        (base / "original_sequences" / "actors" / "c23" / "videos", 0),
        (base / "manipulated_sequences" / "DeepFakeDetection" / "c23" / "videos", 1),
    ]
    records: list[Record] = []
    bad: list[str] = []
    edges: list[tuple[str, str]] = []
    n_found = {0: 0, 1: 0}
    for d, want in dirs:
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() != ".mp4" or not p.is_file():
                bad.append(rel_posix(p, data_root))
                continue
            try:
                info = parse_dfd_name(p.name)
            except ValueError:
                bad.append(rel_posix(p, data_root))
                continue
            if info["label"] != want:
                bad.append(rel_posix(p, data_root))
                continue
            n_found[want] += 1
            if want == 1:
                edges.append((info["target"], info["source"]))
                notes = f"source={info['source']};experiment={info['experiment']}"
                sid = f"{KEY}:manipulated/{p.name}"
                method = "DeepFakeDetection"
            else:
                notes, sid, method = "", f"{KEY}:original/{p.name}", "real"
            records.append(
                Record(sid, KEY, FAMILY, rel_posix(p, data_root), "video", want, method,
                       f"{KEY}:{info['target']}", "identity", True, "", notes)
            )
    actors = sorted({r.group_id.split(":")[1] for r in records} | {s for _, s in edges})
    comp = components(actors, edges)
    comp_sizes = Counter(comp.values())
    real_per = Counter(r.group_id for r in records if r.label == 0)
    fake_per = Counter(r.group_id for r in records if r.label == 1)
    report = {
        "dataset": KEY,
        "originals_found": n_found[0],
        "originals_expected": EXPECTED_ORIGINALS,
        "manipulated_found": n_found[1],
        "manipulated_expected": EXPECTED_MANIPULATED,
        "rows": len(records),
        "actors_total": len(actors),
        "actor_groups_in_manifest": len({r.group_id for r in records}),
        "target_actors_of_fakes": len({t for t, _ in edges}),
        "source_actors_of_fakes": len({s for _, s in edges}),
        "distinct_target_source_pairs": len(set(edges)),
        "fakes_with_target_equal_source": sum(1 for t, s in edges if t == s),
        "connected_components_over_actor_graph": len(comp_sizes),
        "component_sizes": sorted(comp_sizes.values(), reverse=True),
        "originals_per_actor_min_max": [min(real_per.values(), default=0), max(real_per.values(), default=0)],
        "fakes_per_target_actor_min_max": [min(fake_per.values(), default=0), max(fake_per.values(), default=0)],
        "unparsed_or_ignored_files": bad,
        "complete": n_found[0] == EXPECTED_ORIGINALS and n_found[1] == EXPECTED_MANIPULATED and not bad,
    }
    return records, report
