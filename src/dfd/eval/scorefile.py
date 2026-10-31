"""Read and validate the score file (PROTOCOL.md section 2).

One UTF-8 CSV with a header. Required columns: run_id, iteration, seed,
modality, dataset, split, sample_id, group_id, label, score. Optional: logit.
Extra columns are ignored.

Checks: required columns present; no empty text field; seed is an integer;
modality in {visual, audio, image, fused}; split in {train, val, test_in,
test_x, dev_x, post_x}; label 0 or 1; score finite and in [0, 1]; logit finite when given.
post_x (D-094) marks a post-freeze evaluation of a frozen checkpoint on data that was
never read before, run once as its own preregistered experiment; it is not a sealed
split of the freeze and is not in the ledger.
Each run_id has one iteration and one seed. Rows that repeat
(run_id, modality, dataset, split, sample_id) are frame rows: they are
aggregated to one clip row with score = mean of the per-frame scores
(PROTOCOL.md section 3); label and group_id must agree across the frames. The
clip logit is then recomputed from the mean score with the clipping rule. A
blank logit is filled with ln(p / (1 - p)), p clipped to [1e-7, 1 - 1e-7].
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .calibration import logit as _logit, sigmoid

REQUIRED = ("run_id", "iteration", "seed", "modality", "dataset", "split",
            "sample_id", "group_id", "label", "score")
MODALITIES = ("visual", "audio", "image", "fused")
SPLITS = ("train", "val", "test_in", "test_x", "dev_x", "post_x")
FINAL_SPLITS = ("test_in", "test_x")  # guarded and logged
DEV_SPLITS = ("dev_x",)  # development slices (D-035): not final, may be evaluated repeatedly
POST_SPLITS = ("post_x",)  # post-freeze evaluations (D-094): one preregistered scoring run, stored scores re-analysed
TEXT = ("run_id", "iteration", "modality", "dataset", "split", "sample_id", "group_id")


class ScoreFileError(ValueError):
    """The score file is malformed."""


@dataclass
class ScoreTable:
    """Columns as numpy arrays of equal length (clip level after aggregation)."""

    run_id: np.ndarray
    iteration: np.ndarray
    seed: np.ndarray
    modality: np.ndarray
    dataset: np.ndarray
    split: np.ndarray
    sample_id: np.ndarray
    group_id: np.ndarray
    label: np.ndarray
    score: np.ndarray
    logit: np.ndarray
    path: str = ""
    sha256: str = ""
    n_frame_rows: int = 0  # rows that were merged into clip rows
    n_logit_given: int = 0
    warnings: list = field(default_factory=list)

    COLUMNS = ("run_id", "iteration", "seed", "modality", "dataset", "split",
               "sample_id", "group_id", "label", "score", "logit")

    def __len__(self):
        return int(self.score.size)

    def select(self, mask) -> "ScoreTable":
        mask = np.asarray(mask)
        kw = {c: getattr(self, c)[mask] for c in self.COLUMNS}
        return ScoreTable(**kw, path=self.path, sha256=self.sha256,
                          n_frame_rows=self.n_frame_rows, n_logit_given=self.n_logit_given,
                          warnings=list(self.warnings))


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _num(vals, name, rows_hint):
    try:
        return np.array(vals, dtype=float)
    except ValueError:
        for i, v in enumerate(vals):
            try:
                float(v)
            except ValueError:
                raise ScoreFileError(f"column {name!r}: not a number {v!r} (data row {i + 1})") from None
        raise


def read_scores(path) -> ScoreTable:
    path = Path(path)
    sha = file_sha256(path)
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        try:
            header = [h.strip() for h in next(reader)]
        except StopIteration:
            raise ScoreFileError("empty file, header required") from None
        missing = [c for c in REQUIRED if c not in header]
        if missing:
            raise ScoreFileError(f"missing required columns: {missing}")
        pos = {c: header.index(c) for c in header}
        want = list(REQUIRED) + (["logit"] if "logit" in pos else [])
        cols = {c: [] for c in want}
        for ln, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) != len(header):
                raise ScoreFileError(f"line {ln}: {len(row)} fields, header has {len(header)}")
            for c in want:
                cols[c].append(row[pos[c]].strip())
    n = len(cols["score"])
    if n == 0:
        raise ScoreFileError("no data rows")

    for c in TEXT:
        if any(v == "" for v in cols[c]):
            raise ScoreFileError(f"column {c!r} has empty values")
    for c, allowed in (("modality", MODALITIES), ("split", SPLITS)):
        bad = sorted(set(cols[c]) - set(allowed))
        if bad:
            raise ScoreFileError(f"column {c!r}: invalid values {bad}, allowed {list(allowed)}")

    seed = _num(cols["seed"], "seed", n)
    if not np.all(np.isfinite(seed)) or not np.all(seed == np.round(seed)):
        raise ScoreFileError("column 'seed' must hold integers")
    label = _num(cols["label"], "label", n)
    if not np.all((label == 0) | (label == 1)):
        raise ScoreFileError("column 'label' must be 0 or 1")
    score = _num(cols["score"], "score", n)
    if not np.all(np.isfinite(score)) or score.min() < 0.0 or score.max() > 1.0:
        raise ScoreFileError("column 'score' must be finite and within [0, 1]")

    given = np.zeros(n, dtype=bool)
    lg = np.full(n, np.nan)
    if "logit" in cols:
        blank = np.array([v == "" for v in cols["logit"]])
        given = ~blank
        if given.any():
            vals = _num([v for v, b in zip(cols["logit"], blank) if not b], "logit", n)
            if not np.all(np.isfinite(vals)):
                raise ScoreFileError("column 'logit' must be finite when given")
            lg[given] = vals
    lg = np.where(given, lg, _logit(score))

    t = {c: np.array(cols[c], dtype=object) for c in TEXT}
    seed = seed.astype(np.int64)
    label = label.astype(np.int64)

    # a run has one iteration and one seed
    for c in ("iteration", "seed"):
        arr = t[c] if c == "iteration" else seed
        seen = {}
        for r, v in zip(t["run_id"], arr):
            if seen.setdefault(r, v) != v:
                raise ScoreFileError(f"run_id {r!r} has more than one {c}")

    # frame rows to clip rows
    key = np.array(["\x1f".join(k) for k in zip(t["run_id"], t["modality"], t["dataset"],
                                                 t["split"], t["sample_id"])], dtype=object)
    uk, first, inv, cnt = np.unique(key, return_index=True, return_inverse=True, return_counts=True)
    n_frame = 0
    if uk.size < n:
        n_frame = int(n - uk.size)
        if np.any(label != label[first][inv]):
            raise ScoreFileError("a sample has frames with different labels")
        if np.any(t["group_id"] != t["group_id"][first][inv]):
            raise ScoreFileError("a sample has frames with different group_id")
        mean = np.bincount(inv, weights=score, minlength=uk.size) / cnt
        multi = cnt > 1
        lg_first = np.where(multi, _logit(mean), lg[first])
        order = np.argsort(first)  # back to first-appearance order
        pick = first[order]
        score, lg = mean[order], lg_first[order]
        label, seed = label[pick], seed[pick]
        t = {c: t[c][pick] for c in TEXT}
        given = given[pick] & ~multi[order]

    warnings = []
    ok = np.isfinite(lg)
    if ok.any():
        dev = float(np.max(np.abs(sigmoid(lg[ok]) - score[ok])))
        if dev > 1e-3:
            warnings.append(f"sigmoid(logit) differs from score by up to {dev:.3g}")
    return ScoreTable(
        run_id=t["run_id"], iteration=t["iteration"], seed=seed, modality=t["modality"],
        dataset=t["dataset"], split=t["split"], sample_id=t["sample_id"], group_id=t["group_id"],
        label=label, score=score, logit=lg, path=str(path), sha256=sha,
        n_frame_rows=n_frame, n_logit_given=int(given.sum()), warnings=warnings,
    )


def write_scores(path, table_cols: dict) -> None:
    """Write a score file from a dict of equal-length columns (used by tests and tools)."""
    names = list(table_cols)
    n = len(table_cols[names[0]])
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(names)
        for i in range(n):
            w.writerow([table_cols[c][i] for c in names])
