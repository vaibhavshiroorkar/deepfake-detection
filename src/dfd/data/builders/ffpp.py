"""FaceForensics++ c23 manifest builder.

Layout (DATA_LOG.md, seen on the files that have arrived):
  data/ffpp/original_sequences/youtube/c23/videos/<id>.mp4                    id = 000 .. 999
  data/ffpp/manipulated_sequences/<Method>/c23/videos/<target>_<source>.mp4   Method in METHODS

Groups. Nodes are original video ids; every manipulated file <target>_<source> is an edge. A group is a
connected component, so a video, its source and every fake made from either stay together. Identity is not
established for FF++ (DATA_LOG.md), hence ``group_basis = source_video`` and ``identity_known = false``.

Official splits. The public split files (train.json, val.json, test.json; lists of [id_a, id_b] pairs) are
fetched once from the FaceForensics repository into data/_manifests/ffpp_official_splits/. The official
split of a file is that of its original id (the target id for a manipulated file). The report states
whether any connected component spans two official splits.

Files are still arriving while this is written: entries without the .mp4 suffix (temporary download files)
are ignored, and the report gives found against the documented 1,000 originals and 1,000 per method.
With partial data a component is built only from the files present, so a group can be smaller than it will
be when the data is complete.
"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

from ..manifest import Record, rel_posix

KEY = "ffpp"
FAMILY = "ffpp"
METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")
EXPECTED_ORIGINALS = 1000
EXPECTED_PER_METHOD = 1000
SPLIT_BASE_URL = "https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/"
SPLIT_NAMES = ("train", "val", "test")

_ORIG_RE = re.compile(r"^(\d{3})\.mp4$")
_MANIP_RE = re.compile(r"^(\d{3})_(\d{3})\.mp4$")


def ensure_official_splits(split_dir: Path) -> dict[str, list[list[str]]]:
    """Load train/val/test.json from split_dir, fetching each missing file once from the public repo."""
    split_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, list[list[str]]] = {}
    for name in SPLIT_NAMES:
        f = split_dir / f"{name}.json"
        if not f.exists():
            with urllib.request.urlopen(SPLIT_BASE_URL + f"{name}.json", timeout=60) as resp:
                f.write_bytes(resp.read())
        out[name] = json.loads(f.read_text(encoding="utf-8"))
    return out


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, x: str) -> None:
        self.parent.setdefault(x, x)

    def find(self, x: str) -> str:
        self.add(x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def components(originals: list[str], edges: list[tuple[str, str]]) -> dict[str, str]:
    """Map every node id to the group label of its connected component.

    The label is the sorted member ids joined by a hyphen (at most 6 ids), else '<smallest id>-n<size>'.
    """
    uf = UnionFind()
    for o in originals:
        uf.add(o)
    for a, b in edges:
        uf.union(a, b)
    members: dict[str, list[str]] = {}
    for node in list(uf.parent):
        members.setdefault(uf.find(node), []).append(node)
    label: dict[str, str] = {}
    for mem in members.values():
        mem = sorted(mem)
        name = "-".join(mem) if len(mem) <= 6 else f"{mem[0]}-n{len(mem)}"
        for n in mem:
            label[n] = name
    return label


def build(data_root: Path, split_dir: Path | None = None) -> tuple[list[Record], dict]:
    root = data_root / "ffpp"
    orig_dir = root / "original_sequences" / "youtube" / "c23" / "videos"
    split_dir = split_dir or (data_root / "_manifests" / "ffpp_official_splits")
    official = ensure_official_splits(split_dir)
    id_split: dict[str, str] = {}
    pair_of: dict[str, frozenset[str]] = {}
    for split, pairs in official.items():
        for a, b in pairs:
            for i in (a, b):
                id_split[i] = split
                pair_of[i] = frozenset((a, b))

    originals: dict[str, Path] = {}
    ignored: list[str] = []
    if orig_dir.is_dir():
        for p in sorted(orig_dir.iterdir()):
            m = _ORIG_RE.match(p.name)
            if m and p.is_file():
                originals[m.group(1)] = p
            else:
                ignored.append(rel_posix(p, data_root))
    manip: list[tuple[str, str, str, Path]] = []  # method, target, source, path
    found_per_method: dict[str, int] = {}
    for method in METHODS:
        d = root / "manipulated_sequences" / method / "c23" / "videos"
        n = 0
        if d.is_dir():
            for p in sorted(d.iterdir()):
                m = _MANIP_RE.match(p.name)
                if m and p.is_file():
                    manip.append((method, m.group(1), m.group(2), p))
                    n += 1
                else:
                    ignored.append(rel_posix(p, data_root))
        found_per_method[method] = n

    edges = [(t, s) for _, t, s, _ in manip]
    label = components(sorted(originals), edges)

    records: list[Record] = []
    for i, p in sorted(originals.items()):
        records.append(
            Record(f"ffpp:original:{i}", KEY, FAMILY, rel_posix(p, data_root), "video", 0, "real",
                   f"ffpp:g{label[i]}", "source_video", False, id_split.get(i, ""), "")
        )
    for method, t, s, p in manip:
        records.append(
            Record(f"ffpp:{method}:{t}_{s}", KEY, FAMILY, rel_posix(p, data_root), "video", 1, method,
                   f"ffpp:g{label[t]}", "source_video", False, id_split.get(t, ""), f"source={s}")
        )

    comp_splits: dict[str, set[str]] = {}
    comp_members: dict[str, set[str]] = {}
    for node, name in label.items():
        comp_splits.setdefault(name, set()).add(id_split.get(node, "<none>"))
        comp_members.setdefault(name, set()).add(node)
    spanning = {c: sorted(s) for c, s in comp_splits.items() if len(s) > 1}
    not_a_pair = [
        c for c, mem in comp_members.items() if len(mem) > 1 and frozenset(mem) != pair_of.get(next(iter(mem)))
    ]

    report = {
        "dataset": KEY,
        "originals_found": len(originals),
        "originals_expected": EXPECTED_ORIGINALS,
        "manipulated_found": found_per_method,
        "manipulated_expected_per_method": EXPECTED_PER_METHOD,
        "complete": len(originals) == EXPECTED_ORIGINALS
        and all(v == EXPECTED_PER_METHOD for v in found_per_method.values()),
        "rows": len(records),
        "groups": len({r.group_id for r in records}),
        "components_spanning_two_official_splits": spanning,
        "components_spanning_count": len(spanning),
        "components_that_are_not_exactly_one_official_pair": len(not_a_pair),
        "ids_without_official_split": sorted(n for n in label if n not in id_split),
        "ignored_files": ignored,
        "official_split_sizes_pairs": {k: len(v) for k, v in official.items()},
    }
    return records, report
