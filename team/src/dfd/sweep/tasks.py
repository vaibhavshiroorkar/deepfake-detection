"""The three tasks of the I-0 sweep (docs/O007_SWEEP.md, D-036) and where their features and results live.

A feature file is data/_features/<candidate>/<task>/<dataset>_<split>.npz with a .json sidecar, written by the extraction
code of the task. Video features have shape [N, T, D] with a boolean `mask` [N, T] (frames without a face are masked);
image and audio features have shape [N, D]. Every file also holds `label`, `group_id` and `sample_id` per clip or item.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

FEATURES = Path("data/_features")
RESULTS = Path("data/_results")

IMAGE_CANDIDATES = ["IMG-01", "IMG-04", "IMG-07", "IMG-08", "IMG-13", "IMG-15", "IMG-17"]
AUDIO_CANDIDATES = ["AU-1", "AU-4", "AU-6", "AU-8"]


@dataclass(frozen=True)
class Task:
    id: str                    # T-V, T-I or T-A
    experiment: str            # E-001, E-002 or E-003 in docs/EXPERIMENTS.md
    title: str
    candidates: tuple
    train: str                 # dataset_split of the training features
    val: str
    devx: str
    video: bool                # frame-level features with a mask, clip score = mean of frame probabilities (D-024)
    family_a: tuple            # the family of the hypothesis (for example self-supervised or vision-language encoders)
    family_b: tuple            # the comparison family (for example supervised CNNs)
    inconclusive_below: float = 0.55   # lower bound of the best candidate at or below this: inconclusive (O007_SWEEP section 8)
    success_lower: float = 0.60        # criterion (b) of the experiment entry
    reference_needed: bool = False     # E-003 compares also with the from-scratch reference
    extra: dict = field(default_factory=dict)


TASKS = {
    "T-V": Task("T-V", "E-001", "frozen image encoders on face video frames", tuple(IMAGE_CANDIDATES),
                "ffpp_train", "ffpp_val", "celebdf_dev_x", True,
                ("IMG-07", "IMG-08", "IMG-13", "IMG-15", "IMG-17"), ("IMG-01", "IMG-04"), 0.55, 0.60),
    "T-I": Task("T-I", "E-002", "frozen image encoders on generated images", tuple(IMAGE_CANDIDATES),
                "progan_train", "progan_val", "cnn_dev_x", False,
                ("IMG-13", "IMG-15", "IMG-17"), ("IMG-01", "IMG-04"), 0.55, 0.65),
    "T-A": Task("T-A", "E-003", "frozen speech encoders on speech deepfakes", tuple(AUDIO_CANDIDATES),
                "asvspoof_train", "asvspoof_dev", "itw_dev_x", False,
                ("AU-1", "AU-4", "AU-8"), ("AU-6",), 0.55, 0.60, reference_needed=True),
}

# candidates that were considered and are not in the sweep, with the reason (rule 7 and D-038); used by the tables
DROPPED = {
    "IMG-05": "Xception41 (aligned variant): replaced by the original Xception IMG-04 (D-038)",
    "IMG-09": "DINOv2 large: reserve, fine-tuning estimate tight on 12 GB (O007_SWEEP 3.1)",
    "IMG-10..12,19": "DINOv3: manual access gate and acknowledgement duty, excluded on purpose (D-038)",
    "IMG-14": "OpenAI CLIP timm re-upload: fallback only, not needed because IMG-13 was accepted (D-038)",
}


def feature_path(cand: str, task: Task, dataset: str) -> Path:
    return FEATURES / cand / task.id / f"{dataset}.npz"


def expected_files(task: Task) -> list[Path]:
    out = []
    for c in task.candidates:
        for ds in (task.train, task.val, task.devx):
            p = feature_path(c, task, ds)
            out += [p, p.with_suffix(".json")]
    return out


def cost_record_ready(task: Task) -> bool:
    """The section 7 cost record of every candidate of the task exists (the tie rule of D-036 uses it)."""
    import json
    f = RESULTS / "cost_record.json"
    if not f.is_file():
        return False
    rec = json.loads(f.read_text()).get(task.id, {})
    return all(c in rec for c in task.candidates)


def is_ready(task: Task) -> bool:
    ref = RESULTS / task.id / "reference_scores.npz"      # the from-scratch reference (reported, never selected) is part of the table
    return all(p.is_file() for p in expected_files(task)) and cost_record_ready(task) and ref.is_file()
