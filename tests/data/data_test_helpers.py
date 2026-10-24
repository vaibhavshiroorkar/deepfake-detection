"""Synthetic fixtures shared by the tests/data tests (temp directories only; no real data is touched)."""

from __future__ import annotations

import json
from pathlib import Path

from dfd.data.manifest import Record


def touch(path: Path, content: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def make_record(i: int, group: str, label: int, family: str = "fam", dataset: str = "ds",
                official: str = "", path: str | None = None, notes: str = "", method: str | None = None) -> Record:
    return Record(
        sample_id=f"{dataset}:s{i}",
        dataset=dataset,
        family=family,
        path=path or f"{dataset}/s{i}.png",
        modality="image",
        label=label,
        method=method or ("real" if label == 0 else "gen"),
        group_id=f"{dataset}:{group}",
        group_basis="class",
        identity_known=False,
        official_split=official,
        notes=notes,
    )


def synthetic_records(n_groups: int = 30, per_group: int = 6, dataset: str = "ds", family: str = "fam") -> list[Record]:
    recs = []
    i = 0
    for g in range(n_groups):
        for k in range(per_group):
            recs.append(make_record(i, f"g{g:03d}", k % 2, family=family, dataset=dataset))
            i += 1
    return recs


def write_ffpp_fixture(root: Path, pairs: list[tuple[str, str]], methods=("Deepfakes", "Face2Face"),
                       official: dict[str, list[list[str]]] | None = None) -> Path:
    """Create data_root/ffpp/... with originals for every id in pairs and manipulated a_b and b_a per method."""
    ids = sorted({i for p in pairs for i in p})
    for i in ids:
        touch(root / "ffpp" / "original_sequences" / "youtube" / "c23" / "videos" / f"{i}.mp4")
    for m in methods:
        for a, b in pairs:
            for t, s in ((a, b), (b, a)):
                touch(root / "ffpp" / "manipulated_sequences" / m / "c23" / "videos" / f"{t}_{s}.mp4")
    split_dir = root / "_manifests" / "ffpp_official_splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    official = official or {"train": [list(p) for p in pairs], "val": [], "test": []}
    for name, content in official.items():
        (split_dir / f"{name}.json").write_text(json.dumps(content), encoding="utf-8")
    return split_dir


def write_asvspoof_fixture(root: Path, with_extras: bool = True) -> None:
    """Tiny ASVspoof 2019 LA layout: train 2 speakers, dev 1, eval 2 (speaker-disjoint)."""
    la = root / "asvspoof2019_la" / "LA"
    proto = la / "ASVspoof2019_LA_cm_protocols"
    proto.mkdir(parents=True)
    spec = {
        "train": ("ASVspoof2019.LA.cm.train.trn.txt", "ASVspoof2019_LA_train", "T",
                  [("LA_0001", "bonafide", "-"), ("LA_0001", "spoof", "A01"), ("LA_0002", "spoof", "A02"),
                   ("LA_0002", "bonafide", "-")]),
        "dev": ("ASVspoof2019.LA.cm.dev.trl.txt", "ASVspoof2019_LA_dev", "D",
                [("LA_0003", "bonafide", "-"), ("LA_0003", "spoof", "A04")]),
        "eval": ("ASVspoof2019.LA.cm.eval.trl.txt", "ASVspoof2019_LA_eval", "E",
                 [("LA_0004", "bonafide", "-"), ("LA_0004", "spoof", "A07"), ("LA_0005", "spoof", "A19")]),
    }
    n = 1000
    for part, (pf, folder, letter, rows) in spec.items():
        lines = []
        for spk, key, atk in rows:
            n += 1
            fid = f"LA_{letter}_{n}"
            lines.append(f"{spk} {fid} - {atk} {key}")
            touch(la / folder / "flac" / f"{fid}.flac")
        (proto / pf).write_text("\n".join(lines) + "\n", encoding="utf-8")
    if with_extras:
        touch(la / "ASVspoof2019_LA_dev" / "flac" / "LA_D_A1045602.flac")
        touch(la / "ASVspoof2019_LA_eval" / "flac" / "LA_E_A1045603.flac")
