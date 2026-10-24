"""Checkpoint folder format and its checks (D-028).

A checkpoint folder holds `metadata.json` and one `head_<task>.npz` per task (arrays mean, std, w, b of the linear head on the
frozen encoder features). The metadata records the configuration, the preprocess_id of every task, the git commit, manifest and split
IDs, the seed, library versions, the calibration temperature and the validation threshold. `load_checkpoint` refuses a folder with missing
fields, a head file whose SHA-256 differs from the recorded one, or a `preprocess_id` that differs from the one in use now.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dfd.prep import config as P

METADATA_VERSION = "1"
TOP_KEYS = ("metadata_version", "checkpoint_id", "iteration", "git_commit", "protocol", "created", "seed", "library_versions",
            "manifest_ids", "split_ids", "tasks")
TASK_KEYS = ("encoder", "head_file", "head_sha256", "preprocess_id", "preprocess_config", "calibration", "threshold", "train")
ENCODER_KEYS = ("cid", "name", "revision", "dim", "input_size")
TASKS = ("T-V", "T-I", "T-A")


class NoCheckpointError(RuntimeError):
    """There is no checkpoint at the given place."""


class CheckpointError(RuntimeError):
    """The checkpoint exists but breaks the D-028 contract."""


class PreprocessMismatchError(CheckpointError):
    """The checkpoint was trained with a preprocessing configuration other than the one in use."""


@dataclass
class Head:
    mean: np.ndarray
    std: np.ndarray
    w: np.ndarray
    b: float

    def prob_rows(self, X: np.ndarray) -> np.ndarray:
        z = ((np.asarray(X, np.float32) - self.mean) / self.std) @ self.w + self.b
        return 1.0 / (1.0 + np.exp(-z))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_checkpoint(folder: str | Path, meta: dict, heads: dict[str, Head]) -> Path:
    """Write head files and metadata.json; the head hashes are filled in here. `meta` needs every key except head_file/head_sha256."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    meta = json.loads(json.dumps(meta))
    meta.setdefault("metadata_version", METADATA_VERSION)
    for task, h in heads.items():
        f = folder / f"head_{task}.npz"
        np.savez(f, mean=h.mean, std=h.std, w=h.w, b=np.array(h.b))
        meta["tasks"][task]["head_file"] = f.name
        meta["tasks"][task]["head_sha256"] = _sha256(f)
    (folder / "metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return folder / "metadata.json"


def load_checkpoint(folder: str | Path | None, check_preprocess: bool = True) -> tuple[dict, dict[str, Head]]:
    if folder is None or not (Path(folder) / "metadata.json").is_file():
        raise NoCheckpointError(
            f"no checkpoint: {folder!s} has no metadata.json. The real adapter cannot run until the MVP checkpoint (D-040) exists; "
            "use --adapter mock only for interface development, its results are not real.")
    folder = Path(folder)
    try:
        meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise CheckpointError(f"metadata.json is not valid JSON: {e}") from e
    missing = [k for k in TOP_KEYS if k not in meta]
    if missing:
        raise CheckpointError(f"metadata.json lacks {missing}")
    if meta["metadata_version"] != METADATA_VERSION:
        raise CheckpointError(f"metadata_version {meta['metadata_version']!r} is not supported (expected {METADATA_VERSION!r})")
    if not meta["tasks"] or any(t not in TASKS for t in meta["tasks"]):
        raise CheckpointError(f"tasks must be a non-empty subset of {TASKS}")
    heads: dict[str, Head] = {}
    for task, tm in meta["tasks"].items():
        miss = [k for k in TASK_KEYS if k not in tm]
        miss += [f"encoder.{k}" for k in ENCODER_KEYS if k not in tm.get("encoder", {})]
        miss += [f"calibration.{k}" for k in ("method", "temperature", "fitted_on") if k not in tm.get("calibration", {})]
        miss += [f"threshold.{k}" for k in ("value", "how") if k not in tm.get("threshold", {})]
        if miss:
            raise CheckpointError(f"{task}: metadata lacks {miss}")
        hf = folder / tm["head_file"]
        if not hf.is_file():
            raise CheckpointError(f"{task}: head file {tm['head_file']} is missing")
        if _sha256(hf) != tm["head_sha256"]:
            raise CheckpointError(f"{task}: head file hash differs from the recorded one (file changed or corrupt)")
        if check_preprocess:
            now = P.IN_USE[task]()
            if tm["preprocess_id"] != now:
                raise PreprocessMismatchError(
                    f"{task}: the checkpoint was trained with preprocess_id {tm['preprocess_id']} but the one in use is {now}; "
                    "refusing to score (D-028 item 4). Retrain, or restore the matching preprocessing configuration.")
        z = np.load(hf, allow_pickle=False)
        heads[task] = Head(z["mean"].astype(np.float32), z["std"].astype(np.float32), z["w"].astype(np.float32), float(z["b"]))
        if heads[task].w.shape[0] != int(tm["encoder"]["dim"]):
            raise CheckpointError(f"{task}: head width {heads[task].w.shape[0]} differs from encoder.dim {tm['encoder']['dim']}")
    return meta, heads


def config_hash_of(meta: dict) -> str:
    return P.config_hash({t: {k: v for k, v in tm.items() if k in ("encoder", "preprocess_config", "head_sha256")} for t, tm in meta["tasks"].items()})
