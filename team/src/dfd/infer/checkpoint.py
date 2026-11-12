"""Checkpoint folder format and its checks (D-028).

Version 1 (the I-0 MVP): a folder holds `metadata.json` and one `head_<task>.npz` per task (arrays mean, std, w, b of the linear head on the
frozen encoder features). The metadata records the configuration, the preprocess_id of every task, the git commit, manifest and split
IDs, the seed, library versions, the calibration temperature and the validation threshold. `load_checkpoint` refuses a folder with missing
fields, a head file whose SHA-256 differs from the recorded one, or a `preprocess_id` that the code cannot apply exactly as trained.

Version 2 adds, per task, a `model` block that says what the task's scorer is:
  {"kind": "linear_probe"}                       one frozen encoder and the head file of version 1 (head_file, head_sha256 as before);
  {"kind": "finetuned_encoder", "weights_file"}  the encoder of `encoder.cid` with the fine-tuned parameters and the linear head stored in one
                                                 torch file (the format written by dfd.train.finetune: keys `params` and `head`);
  {"kind": "ensemble_logit_mean", "members"}     several frozen encoders, each with its own head file; the score is the mean of the members' logits.
and a top-level `files` map of relative path to SHA-256 for every file of the folder except metadata.json. Every listed file must exist and match,
and every file the model blocks name must be listed. A top-level `fusion` key, if present, must be null: audio-visual fusion is not implemented
(extension point in dfd.infer.real_adapter, `AUDIO_VISUAL_FUSION`).

Preprocessing (D-028 item 4): the recorded `preprocess_id` must be the hash of the recorded `preprocess_config` and must be one the code can apply
exactly as trained (dfd.prep.config.ACCEPTED). The adapter then applies the preprocessing the checkpoint records, so a version 1 checkpoint is scored
with the first-window audio path it was trained with, and an id the code does not know is refused with PreprocessMismatchError.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from dfd.prep import config as P

SUPPORTED_VERSIONS = ("1", "2")
METADATA_VERSION = "1"            # the default of save_checkpoint (version 1 writer); save_checkpoint_v2 writes "2"
TOP_KEYS = ("metadata_version", "checkpoint_id", "iteration", "git_commit", "protocol", "created", "seed", "library_versions",
            "manifest_ids", "split_ids", "tasks")
TASK_KEYS_V1 = ("encoder", "head_file", "head_sha256", "preprocess_id", "preprocess_config", "calibration", "threshold", "train")
TASK_KEYS_V2 = ("encoder", "preprocess_id", "preprocess_config", "calibration", "threshold", "train", "model")
TASK_KEYS = TASK_KEYS_V1
ENCODER_KEYS = ("cid", "name", "revision", "dim", "input_size")
MODEL_KINDS = ("linear_probe", "finetuned_encoder", "ensemble_logit_mean")
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

    def logit_rows(self, X: np.ndarray) -> np.ndarray:
        return ((np.asarray(X, np.float32) - self.mean) / self.std) @ self.w + self.b

    def prob_rows(self, X: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.logit_rows(X)))


@dataclass
class FineTunedModel:
    """A fine-tuned encoder and its head, stored in one torch file; loaded lazily by the adapter."""
    weights_path: Path
    dim: int


@dataclass
class Member:
    cid: str
    encoder: dict
    head: Head


@dataclass
class Ensemble:
    """Frozen encoders with a linear head each; the task score is the mean of the members' logits."""
    members: list[Member] = field(default_factory=list)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_checkpoint(folder: str | Path, meta: dict, heads: dict[str, Head]) -> Path:
    """Write head files and metadata.json (version 1); the head hashes are filled in here. `meta` needs every key except head_file/head_sha256."""
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


def save_head(path: str | Path, h: Head) -> None:
    np.savez(path, mean=h.mean, std=h.std, w=h.w, b=np.array(h.b))


def save_checkpoint_v2(folder: str | Path, meta: dict, artefacts: dict[str, Path | Head] | None = None) -> Path:
    """Write a version 2 checkpoint. `artefacts` maps a file name in the folder to a source path (copied) or to a Head (written as npz); the model
    blocks of `meta` name these files. The `files` map (SHA-256 of every file but metadata.json) is computed here."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, src in (artefacts or {}).items():
        if isinstance(src, Head):
            save_head(folder / name, src)
        else:
            shutil.copyfile(src, folder / name)
    meta = json.loads(json.dumps(meta))
    meta["metadata_version"] = "2"
    meta.setdefault("fusion", None)
    meta["files"] = {p.name: _sha256(p) for p in sorted(folder.iterdir()) if p.is_file() and p.name != "metadata.json"}
    (folder / "metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return folder / "metadata.json"


def _check_files(folder: Path, meta: dict) -> None:
    files = meta.get("files")
    if not isinstance(files, dict) or not files:
        raise CheckpointError("version 2 metadata lacks the `files` map of SHA-256 values")
    for name, digest in files.items():
        f = folder / name
        if not f.is_file():
            raise CheckpointError(f"file {name} listed in the metadata is missing")
        if _sha256(f) != digest:
            raise CheckpointError(f"file {name}: hash differs from the recorded one (file changed or corrupt)")
    listed = set(files)
    for task, tm in meta["tasks"].items():
        mb = tm["model"]
        named = [mb["weights_file"]] if mb["kind"] == "finetuned_encoder" else [m["head_file"] for m in mb["members"]] if mb["kind"] == "ensemble_logit_mean" else [tm["head_file"]]
        for n in named:
            if n not in listed:
                raise CheckpointError(f"{task}: file {n} named by the model block is not in the `files` map")


def _load_head(path: Path, dim: int, label: str) -> Head:
    z = np.load(path, allow_pickle=False)
    h = Head(z["mean"].astype(np.float32), z["std"].astype(np.float32), z["w"].astype(np.float32), float(z["b"]))
    if h.w.shape[0] != int(dim):
        raise CheckpointError(f"{label}: head width {h.w.shape[0]} differs from encoder.dim {dim}")
    return h


def load_checkpoint(folder: str | Path | None, check_preprocess: bool = True) -> tuple[dict, dict]:
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
    version = meta["metadata_version"]
    if version not in SUPPORTED_VERSIONS:
        raise CheckpointError(f"metadata_version {version!r} is not supported (expected one of {SUPPORTED_VERSIONS})")
    if not meta["tasks"] or any(t not in TASKS for t in meta["tasks"]):
        raise CheckpointError(f"tasks must be a non-empty subset of {TASKS}")
    v2 = version == "2"
    if v2 and meta.get("fusion") is not None:
        raise CheckpointError("the metadata names an audio-visual fusion rule, which this library does not implement")
    heads: dict = {}
    for task, tm in meta["tasks"].items():
        kind = tm.get("model", {}).get("kind", "linear_probe") if v2 else "linear_probe"
        if v2 and kind not in MODEL_KINDS:
            raise CheckpointError(f"{task}: unknown model kind {kind!r}")
        need = TASK_KEYS_V2 if v2 else TASK_KEYS_V1
        miss = [k for k in need if k not in tm]
        if v2 and kind == "linear_probe":
            miss += [k for k in ("head_file",) if k not in tm]
        miss += [f"encoder.{k}" for k in ENCODER_KEYS if k not in tm.get("encoder", {})]
        miss += [f"calibration.{k}" for k in ("method", "temperature", "fitted_on") if k not in tm.get("calibration", {})]
        miss += [f"threshold.{k}" for k in ("value", "how") if k not in tm.get("threshold", {})]
        if v2 and kind == "finetuned_encoder" and "weights_file" not in tm["model"]:
            miss.append("model.weights_file")
        if v2 and kind == "ensemble_logit_mean" and not tm["model"].get("members"):
            miss.append("model.members")
        if miss:
            raise CheckpointError(f"{task}: metadata lacks {miss}")
        if check_preprocess:
            pid = tm["preprocess_id"]
            if pid != P.config_hash(tm["preprocess_config"]):
                raise PreprocessMismatchError(f"{task}: preprocess_id {pid} is not the hash of the recorded preprocess_config")
            if pid not in P.ACCEPTED[task]():
                raise PreprocessMismatchError(
                    f"{task}: the checkpoint was trained with preprocess_id {pid} but the ones this code can apply are {sorted(P.ACCEPTED[task]())}; "
                    "refusing to score (D-028 item 4). Retrain, or restore the matching preprocessing configuration.")
    if v2:
        _check_files(folder, meta)
    for task, tm in meta["tasks"].items():
        kind = tm["model"]["kind"] if v2 else "linear_probe"
        if kind == "linear_probe":
            hf = folder / tm["head_file"]
            if not hf.is_file():
                raise CheckpointError(f"{task}: head file {tm['head_file']} is missing")
            if "head_sha256" in tm and _sha256(hf) != tm["head_sha256"]:
                raise CheckpointError(f"{task}: head file hash differs from the recorded one (file changed or corrupt)")
            heads[task] = _load_head(hf, tm["encoder"]["dim"], task)
        elif kind == "finetuned_encoder":
            wf = folder / tm["model"]["weights_file"]
            if not wf.is_file():
                raise CheckpointError(f"{task}: weights file {tm['model']['weights_file']} is missing")
            heads[task] = FineTunedModel(wf, int(tm["encoder"]["dim"]))
        else:
            members = []
            for m in tm["model"]["members"]:
                miss = [f"member.{k}" for k in (*ENCODER_KEYS, "head_file") if k not in m]
                if miss:
                    raise CheckpointError(f"{task}: metadata lacks {miss}")
                hf = folder / m["head_file"]
                if not hf.is_file():
                    raise CheckpointError(f"{task}: head file {m['head_file']} is missing")
                members.append(Member(m["cid"], {k: m[k] for k in ENCODER_KEYS}, _load_head(hf, m["dim"], f"{task} {m['cid']}")))
            heads[task] = Ensemble(members)
    return meta, heads


def config_hash_of(meta: dict) -> str:
    d = {t: {k: v for k, v in tm.items() if k in ("encoder", "preprocess_config", "head_sha256", "model")} for t, tm in meta["tasks"].items()}
    if "files" in meta:
        d["_files"] = meta["files"]
    return P.config_hash(d)
