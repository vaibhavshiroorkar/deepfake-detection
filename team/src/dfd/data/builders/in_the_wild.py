"""In-the-Wild audio manifest builder (test_x with a dev_x slice carved out by splits.py).

Layout (verified on the real folder, 2026-10-24):
  data/in_the_wild/release_in_the_wild/<n>.wav      31,779 wav files
  data/in_the_wild/release_in_the_wild/meta.csv     header ``file,speaker,label``; label values ``bona-fide`` and ``spoof``
The folder also holds one non-wav file (not listed in meta.csv, ignored and counted in the report).

Rules. label bona-fide -> 0 (method ``real``), spoof -> 1 (method ``spoof``: the generator is not given).
Group = the speaker name (identity_known true, group_basis identity). Every speaker has both labels in the
real file, so a speaker group holds real and fake clips. Family ``in_the_wild``. No official split.
Any other label value, a meta row whose wav is missing, or a duplicate file name raises ValueError.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

from ..manifest import Record, rel_posix

KEY = "in_the_wild"
FAMILY = "in_the_wild"
LABELS = {"bona-fide": 0, "spoof": 1}
DOCUMENTED_SPEAKERS = 58  # HF dataset card, DATA_LOG.md A4


def build(data_root: Path) -> tuple[list[Record], dict]:
    base = data_root / KEY / "release_in_the_wild"
    meta = base / "meta.csv"
    if not meta.is_file():
        raise FileNotFoundError(f"meta.csv not found: {meta}")
    on_disk = {p.name for p in base.iterdir() if p.is_file()}
    records: list[Record] = []
    seen: set[str] = set()
    spk_labels: dict[str, Counter] = defaultdict(Counter)
    with open(meta, "r", encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh)
        if rd.fieldnames != ["file", "speaker", "label"]:
            raise ValueError(f"meta.csv header {rd.fieldnames} != ['file', 'speaker', 'label']")
        for row in rd:
            f, spk, lab = row["file"], row["speaker"], row["label"]
            if lab not in LABELS:
                raise ValueError(f"unknown label {lab!r} in meta.csv row {row}")
            if f in seen:
                raise ValueError(f"duplicate file {f} in meta.csv")
            if f not in on_disk:
                raise ValueError(f"meta.csv lists {f} but it is not on disk")
            if not spk.strip():
                raise ValueError(f"empty speaker in meta.csv row {row}")
            seen.add(f)
            label = LABELS[lab]
            spk_labels[spk][lab] += 1
            records.append(
                Record(f"{KEY}:{f}", KEY, FAMILY, rel_posix(base / f, data_root), "audio", label,
                       "real" if label == 0 else "spoof", f"{KEY}:{spk}", "identity", True, "", "")
            )
    wavs = {n for n in on_disk if n.lower().endswith(".wav")}
    report = {
        "dataset": KEY,
        "rows": len(records),
        "label_counts": dict(Counter(r.label for r in records)),
        "meta_label_values": dict(Counter(LABELS_INV[r.label] for r in records)),
        "speakers": len(spk_labels),
        "speakers_documented": DOCUMENTED_SPEAKERS,
        "speakers_with_both_labels": sum(1 for c in spk_labels.values() if len(c) == 2),
        "clips_per_speaker_min_max": [min(sum(c.values()) for c in spk_labels.values()),
                                      max(sum(c.values()) for c in spk_labels.values())] if spk_labels else [0, 0],
        "wav_files_on_disk": len(wavs),
        "wav_files_not_in_meta": len(wavs - seen),
        "non_wav_files_ignored": sorted(on_disk - wavs),
        "complete": len(records) > 0 and len(wavs - seen) == 0,
    }
    return records, report


LABELS_INV = {v: k for k, v in LABELS.items()}
