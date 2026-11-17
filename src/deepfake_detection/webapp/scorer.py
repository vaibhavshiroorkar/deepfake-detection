"""Score one uploaded clip with the three P-001 checks, in this process.

Every step calls the code the study used, so an upload is scored as the paper's
test clips were:

1. neutralizer v4 (`preprocessing.neutralize.neutralize_clip`), native frame size;
2. the P-001 preprocessor (`code_version="p001"`), refused if its hash is not the
   one every P-001 model was trained on;
3. face and voice: the six visual and audio checkpoints through `CachedBranchDataset`,
   exactly as `ddf evaluate branch` reads a cached clip;
4. lip sync: mouth crops and frozen AV-HuBERT for the 2 s windows at 0 s and 1 s
   (P-001 amendment A4), stored as float16 like the feature cache, then the three
   alignment heads; a window needs a face in 80% of its 50 frames.

A check that cannot run returns no score and a reason. The models load once, in
`Scorer.__init__`; a cold start takes about two minutes on two CPU threads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from deepfake_detection.paths import PROJECT_ROOT
from deepfake_detection.webapp.calibration import check_score

P001_PREPROCESSING_HASH = "5f2954fbda2cd643b4da48b8a526b0aa3d9ad831033ac8d22704d78d885df7d1"
# The released AV-HuBERT Base checkpoint. Its loader unpickles with weights_only=False, so a changed
# file could run code: it is refused unless it is exactly this file (docs/SECURITY.md).
AVHUBERT_SHA256 = "dee8b0651dbea3a21cda0f2c52b956e3a878d5f6ee503d427d6a4044bb95c9cc"
SEEDS = (17, 29, 43)
SYNC_STARTS = (0.0, 1.0)        # P-001 amendment A4: two 2 s windows of a 3.00 s neutral clip
MIN_FACE = 0.8                  # as scripts/train_sync_path.py
WINDOW_FRAMES = 50
HOP = 640                       # 16 kHz samples per 25 fps frame
BRANCH_VIEW = {"face": ("visual", "visual_view"), "voice": ("audio", "audio_view")}


def verified_avhubert(path: Path) -> Path:
    """`path`, after checking it is the released checkpoint. Raises before anything is unpickled."""
    from deepfake_detection.digest import file_sha256

    if file_sha256(path) != AVHUBERT_SHA256:
        raise RuntimeError(f"{path} does not have the released AV-HuBERT SHA-256; refusing to load it")
    return path


class ClipRefused(ValueError):
    """The clip cannot be checked at all, with a reason a visitor can read."""


@dataclass(frozen=True)
class CheckResult:
    probabilities: tuple[float, ...] | None      # one per seed, or None when the check could not run
    reason: str | None = None

    @property
    def score(self) -> float | None:
        return None if self.probabilities is None else check_score(list(self.probabilities))


@dataclass(frozen=True)
class ScoreReport:
    checks: dict[str, CheckResult]
    trace: dict = field(default_factory=dict)

    def scores(self) -> dict[str, float | None]:
        return {name: result.score for name, result in self.checks.items()}


@dataclass
class _Models:
    preprocessor: object
    branches: dict[str, list[torch.nn.Module]]
    avhubert: torch.nn.Module
    heads: list[torch.nn.Module]


class Scorer:
    def __init__(self, root: Path = PROJECT_ROOT, device: str = "cpu") -> None:
        from deepfake_detection.avsync.features import init_worker
        from deepfake_detection.avsync.heads import AlignmentHead
        from deepfake_detection.branches.registry import load_branch
        from deepfake_detection.inference.loading import build_preprocessor
        from deepfake_detection.streams.avhubert import load_avhubert

        self.device = device
        checkpoints = root / "runs" / "p001" / "checkpoints"
        branches = {check: [load_branch(checkpoints / f"{branch}-seed{seed}.pt", branch)[0].to(device).eval()
                            for seed in SEEDS] for check, (branch, _) in BRANCH_VIEW.items()}
        heads = []
        for seed in SEEDS:
            state = torch.load(root / "runs" / "p001" / "sync" / f"sync-seed{seed}.pt", map_location="cpu",
                               weights_only=True)
            head = AlignmentHead(width=768)
            head.load_state_dict(state["state_dict"])
            heads.append(head.to(device).eval())
        init_worker(str(root / "models" / "mediapipe" / "face_landmarker.task"), 64.0)
        self.models = _Models(
            preprocessor=build_preprocessor(code_version="p001", device=device),
            branches=branches,
            avhubert=load_avhubert(verified_avhubert(root / "models" / "avhubert" / "base_vox_iter5.pt")).to(device).eval(),
            heads=heads,
        )

    def score(self, upload: Path, workdir: Path, progress=lambda stage: None) -> ScoreReport:
        """Score `upload`, writing every derived file under `workdir`. The caller deletes `workdir`.

        `progress` gets the stage names the page shows: cleaning, finding the face, scoring.
        """
        progress("cleaning")
        neutral = self._neutralize(upload, workdir / "neutral.mp4")
        progress("finding the face")
        checks = self._face_and_voice(neutral, workdir / "cache")
        progress("scoring")
        sync, trace = self._sync(neutral)
        return ScoreReport({**checks, "sync": sync}, trace)

    @staticmethod
    def _neutralize(upload: Path, target: Path) -> Path:
        from deepfake_detection.preprocessing.neutralize import (
            NeutralSettings,
            neutralize_clip,
        )

        report = neutralize_clip(upload, target, NeutralSettings())
        if "error" in report:
            if str(report["error"]).startswith("too short"):
                raise ClipRefused("The clip has less than 3 seconds of sound, so it cannot be cleaned and checked.")
            raise RuntimeError(f"neutralizer failed: {report['error']}")
        return target

    def _face_and_voice(self, neutral: Path, cache_root: Path) -> dict[str, CheckResult]:
        from deepfake_detection.data.datasets import (
            CachedBranchDataset,
            collate_branch_items,
        )
        from deepfake_detection.data.manifest import ClipRecord
        from deepfake_detection.training.input_contract import audio_forward
        from deepfake_detection.views.cache_store import CacheStore

        # An upload has no label. ClipRecord requires a consistent one, so it gets the real
        # placeholder; the label only feeds the loss in training and is never read here.
        record = ClipRecord(clip_id="upload", dataset="Upload", video_path=neutral,
                            manipulation_type="RealVideo-RealAudio", method="upload", source="upload", targets=(),
                            clip_fake=False, video_fake=False, audio_fake=False)
        prepared = self.models.preprocessor.prepare(record, neutral)
        if prepared.preprocessing_config_hash != P001_PREPROCESSING_HASH:
            raise RuntimeError(f"preprocessing hash {prepared.preprocessing_config_hash} is not the P-001 hash")
        store = CacheStore(cache_root)
        path = store.save(prepared, dataset=record.dataset)
        views = store.available_views(path)
        out = {}
        for check, (branch, view) in BRANCH_VIEW.items():
            if view not in views:
                out[check] = CheckResult(None, "no steady face was found" if check == "face" else "no usable sound")
                continue
            item = CachedBranchDataset(records=[record], cache_index={record.clip_id: path}, cache_store=store,
                                       branch=branch, preprocessing_hash=P001_PREPROCESSING_HASH)[0]
            batch = collate_branch_items([item])
            probs = []
            with torch.inference_mode():
                for model in self.models.branches[check]:
                    logits = audio_forward(model, batch.values.to(self.device), batch.valid_samples).logits
                    probs.append(float(torch.sigmoid(logits).reshape(-1)[0]))
            out[check] = CheckResult(tuple(probs))
        return out

    def _sync(self, neutral: Path) -> tuple[CheckResult, dict]:
        from deepfake_detection.avsync.features import encode, prepare_window

        windows, loudness = [], None
        for start in SYNC_STARTS:
            face, crops, audio = prepare_window((str(neutral), start))
            if loudness is None and audio is not None and len(audio):
                loudness = _loudness_db(audio)
            if isinstance(face, str):
                raise RuntimeError(f"sync window failed: {face}")
            if crops is None or audio is None or not len(audio) or face < MIN_FACE:
                continue
            features = encode(self.models.avhubert, crops, audio, device=self.device)
            lips, sound = features["lips"], features["audio"]
            if lips.shape[0] != WINDOW_FRAMES or sound.shape[0] != WINDOW_FRAMES:
                continue
            windows.append((start, lips.astype(np.float32), sound.astype(np.float32)))
        trace = {"loudness_db": loudness or [], "windows": []}
        if not windows:
            return CheckResult(None, "the mouth or the sound could not be followed for 2 seconds"), trace
        lips = torch.from_numpy(np.stack([w[1] for w in windows])).to(self.device)
        sound = torch.from_numpy(np.stack([w[2] for w in windows])).to(self.device)
        per_seed = []
        with torch.no_grad():
            for head in self.models.heads:
                per_seed.append(head.window_score(lips, sound).cpu().numpy())
        misfit = np.stack(per_seed)                      # [seeds, windows]
        probabilities = tuple(float(1 - math.exp(-s)) for s in misfit.mean(axis=1))
        trace["windows"] = [{"start": w[0], "misfit": float(1 - math.exp(-m))}
                            for w, m in zip(windows, misfit.mean(axis=0), strict=True)]
        return CheckResult(probabilities), trace


def _loudness_db(audio: np.ndarray) -> list[float]:
    """Speech loudness in dB per 40 ms frame, for the trace."""
    frames = len(audio) // HOP
    rms = np.sqrt((audio[: frames * HOP].reshape(frames, HOP) ** 2).mean(axis=1) + 1e-10)
    return [round(float(v), 2) for v in 20 * np.log10(rms)]
