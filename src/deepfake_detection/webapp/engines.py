"""The models the app can serve, loaded once and on first use.

Three choices, because the right one depends on the footage and no single model
wins everywhere. The ensemble ranks best on DFDC, the single model is what most
of the project's findings were measured on, and the multimodal fusion is the
only one that hears the audio.
"""

from __future__ import annotations

import base64
import io
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from deepfake_detection.paths import PROJECT_ROOT

PROGRAM_CHECKPOINTS = PROJECT_ROOT / "runs" / "program-20260906" / "checkpoints"
STREAM_RUN = PROJECT_ROOT / "runs" / "design-b-20260910"
DENSITY_PATH = PROJECT_ROOT / "runs" / "probe-20260914" / "authenticity-density.npz"
# Both the Design A visual checkpoints and the Design B streams were trained on
# crops built by this code version, and the loaders check it against each
# checkpoint's preprocessing hash.
CODE_VERSION = "program-v1"
BEST_SINGLE = "efficientnet-b0-seed17"


@dataclass(frozen=True, slots=True)
class ModelInfo:
    id: str
    label: str
    summary: str
    media: tuple[str, ...]
    # Evaluation report stems under runs/, read by `results.model_metrics`.
    reports: dict[str, str] = field(default_factory=dict)


MODELS: tuple[ModelInfo, ...] = (
    ModelInfo(
        id="ensemble",
        label="Visual ensemble",
        summary=(
            "Six face models (EfficientNet-B0 and ConvNeXt-Tiny, three seeds "
            "each), averaged. Best on DFDC, level with the single model on "
            "Celeb-DF, and it catches fewer MNW fakes at 0.5."
        ),
        media=("video",),
        reports={
            "FakeAVCeleb": "program-20260906/evaluation/visual-ensemble-in-domain-test",
            "DFDC": "program-20260906/evaluation/visual-ensemble-dfdc",
            "Celeb-DF-v2": "program-20260906/evaluation/visual-ensemble-celebdf",
        },
    ),
    ModelInfo(
        id="visual-best",
        label="Single face model",
        summary=(
            "EfficientNet-B0 with a GRU over 16 face crops, seed 17. The "
            "baseline most findings were measured against."
        ),
        media=("video",),
        reports={
            "FakeAVCeleb": "program-20260906/evaluation/visual-in-domain-test-seed17",
            "DFDC": "program-20260906/evaluation/visual-dfdc-seed17",
            "Celeb-DF-v2": "program-20260906/evaluation/visual-celebdf",
        },
    ),
    ModelInfo(
        id="multimodal",
        label="Audio-visual fusion",
        summary=(
            "Five streams (two face backbones, voice, lip-sync, emotion) fused "
            "by a learned head. Also reads audio-only and image files. Near "
            "perfect in-domain, close to chance on DFDC."
        ),
        media=("video", "image", "audio"),
        reports={
            "FakeAVCeleb": "design-b-20260910/evaluation/gate-fusion-in-domain",
            "DFDC": "design-b-20260910/evaluation/gate-fusion-dfdc",
        },
    ),
)
MODEL_IDS = tuple(model.id for model in MODELS)


@dataclass(frozen=True, slots=True)
class Analysis:
    model: str
    verdict: str
    probability: float | None
    threshold: float
    members: dict[str, float]
    blockers: tuple[str, ...]
    familiarity: dict[str, Any] | None
    withheld: bool
    faces: tuple[str, ...]
    media_kind: str
    seconds: float


def face_thumbnails(view: np.ndarray | None, *, count: int = 8) -> tuple[str, ...]:
    """A few of the crops the model saw, as data URLs.

    The view is ImageNet-normalised CHW floats, so it is undone before encoding,
    the same arithmetic the dashboard's `display_face_frames` uses.
    """
    if view is None:
        return ()
    from PIL import Image

    values = np.asarray(view, dtype=np.float32)
    mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32).reshape(1, 3, 1, 1)
    deviation = np.asarray((0.229, 0.224, 0.225), dtype=np.float32).reshape(1, 3, 1, 1)
    pixels = np.clip((values * deviation + mean) * 255.0, 0, 255).astype(np.uint8)
    frames = pixels.transpose(0, 2, 3, 1)
    picks = np.linspace(0, len(frames) - 1, num=min(count, len(frames))).astype(int)
    urls = []
    for index in picks:
        buffer = io.BytesIO()
        Image.fromarray(frames[index]).save(buffer, format="JPEG", quality=85)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        urls.append(f"data:image/jpeg;base64,{encoded}")
    return tuple(urls)


class Registry:
    """Lazy, thread-safe holders for the engines and the familiarity gate.

    Loading the multimodal engine reads about 3.7 GB, so nothing loads until a
    request needs it, and a second request waits for the first load rather than
    starting its own.
    """

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self._lock = threading.Lock()
        self._loaded: dict[str, Any] = {}
        self._errors: dict[str, str] = {}

    def _get(self, key: str, build: Callable[[], Any]) -> Any:
        with self._lock:
            if key not in self._loaded:
                try:
                    self._loaded[key] = build()
                except Exception as error:
                    self._errors[key] = str(error)
                    raise
                self._errors.pop(key, None)
            return self._loaded[key]

    def status(self) -> dict[str, str]:
        state = {key: "loaded" for key in self._loaded}
        state.update({key: f"failed: {error}" for key, error in self._errors.items()})
        return state

    def ensemble(self):
        def build():
            from deepfake_detection.inference.ensemble import (
                load_visual_ensemble,
                program_members,
            )

            return load_visual_ensemble(
                program_members(PROGRAM_CHECKPOINTS),
                code_version=CODE_VERSION,
                device=self.device,
            )

        return self._get("ensemble", build)

    def multimodal(self):
        def build():
            from deepfake_detection.inference.multimodal import load_multimodal_engine

            # The gate runs here for every model, so the engine is built
            # without its own copy.
            return load_multimodal_engine(
                run_dir=STREAM_RUN,
                code_version=CODE_VERSION,
                device=self.device,
                root=PROJECT_ROOT,
                with_familiarity=False,
            )

        return self._get("multimodal", build)

    def familiarity(self):
        def build():
            from deepfake_detection.inference.authenticity import load_gate

            return load_gate(DENSITY_PATH, device=self.device)

        return self._get("familiarity", build)

    def analyze(
        self,
        path: Path,
        *,
        model: str,
        threshold: float | None,
        withhold_unfamiliar: bool,
        default_threshold: float = 0.5,
    ) -> Analysis:
        """Score one file.

        `threshold` is the user's own cut-off, or None when they left it alone.
        Then a video gets `default_threshold`, and an image or a sound file
        gets the cut-off the fusion head was fitted with for that media kind,
        because one threshold cannot serve three media kinds (F7).
        """
        if model not in MODEL_IDS:
            raise ValueError(f"Unknown model {model!r}")
        for value in (threshold, default_threshold):
            if value is not None and not 0 <= value <= 1:
                raise ValueError("Threshold must be in [0, 1]")
        from deepfake_detection import media_kind

        started = time.perf_counter()
        kind = media_kind.classify(path)
        familiarity = None
        gate_unavailable = False
        if kind == media_kind.VIDEO:
            # A gate that is missing or fails to load must not quietly turn
            # into "familiar": the verdict says the check did not run.
            try:
                gate = self.familiarity()
            except Exception:
                gate = None
            check = gate.check_clip(path) if gate is not None else None
            gate_unavailable = check is None
            if check is not None:
                familiarity = {
                    "familiar": check.familiar,
                    "distance": check.distance,
                    "threshold": check.threshold,
                    "reason": check.reason,
                }

        if model == "multimodal":
            engine = self.multimodal()
            result = engine.predict(path)
            if threshold is None and kind != media_kind.VIDEO and result.threshold:
                threshold = result.threshold
            members = {
                name: float(1 / (1 + np.exp(-logit)))
                for name, logit in result.branch_logits.items()
            }
            faces: tuple[str, ...] = ()
            probability = result.probability
            blockers = result.blockers
        else:
            if kind != media_kind.VIDEO:
                raise ValueError(
                    "The face models read video only. Choose the audio-visual "
                    "fusion for images and audio files."
                )
            engine = self.ensemble()
            subset = None if model == "ensemble" else (BEST_SINGLE,)
            prediction = engine.predict(
                path,
                threshold=default_threshold if threshold is None else threshold,
                members=subset,
            )
            result = prediction.result
            members = prediction.member_probabilities
            faces = face_thumbnails(prediction.visual_view)
            probability = result.probability
            blockers = result.blockers

        if threshold is None:
            threshold = default_threshold
        unfamiliar = familiarity is not None and not familiarity["familiar"]
        if unfamiliar:
            blockers = (*blockers, "unfamiliar_capture_conditions")
        if gate_unavailable and withhold_unfamiliar:
            blockers = (*blockers, "familiarity_check_unavailable")
        withheld = bool(unfamiliar and withhold_unfamiliar)
        if probability is None:
            verdict = "indeterminate"
        elif withheld:
            verdict = "withheld"
        else:
            verdict = "fake" if probability >= threshold else "real"
        return Analysis(
            model=model,
            verdict=verdict,
            probability=probability,
            threshold=threshold,
            members=members,
            blockers=tuple(dict.fromkeys(blockers)),
            familiarity=familiarity,
            withheld=withheld,
            faces=faces,
            media_kind=kind,
            seconds=time.perf_counter() - started,
        )
