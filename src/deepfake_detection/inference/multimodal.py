"""Score a video, a still image or a sound file through whatever streams it drives.

`PredictionEngine` next door runs three fixed branches and then demands all
three before it will issue a verdict, so a clip with partial coverage always
comes back indeterminate. That is right for the frozen Design A system, where
fusion cannot consume a subset. The feature-level head can, so this engine runs
what the input supports and fuses what ran.

The routing is not decided here. `dashboard/lib/media_kind.py` owns the table of
which streams a media kind can drive, and this asks it, so the serving path and
the panel the user sees can never disagree about what will run.

A clip that drives nothing still returns a result: verdict "indeterminate",
probability None, and the blockers that explain it. Silence would be worse than
a refusal, and the abstention rate the data card requires is counted from
exactly these.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from deepfake_detection.dashboard.lib import media_kind
from deepfake_detection.fusion.deep_loading import LoadedFusion, fuse
from deepfake_detection.fusion.stream_export import StreamSpec, _forward
from deepfake_detection.inference.predictor import PredictionResult


@dataclass(frozen=True, slots=True)
class _Record:
    """The minimal record the preprocessor needs for an ad-hoc file."""

    clip_id: str
    dataset: str = "upload"
    leading_silence_sec: float = 0.0
    sync_start_sec: float = 0.0


class MultimodalEngine:
    """Runs the streams an input supports, then the fusion head over them."""

    def __init__(
        self,
        *,
        preprocessor: Any,
        streams: dict[str, StreamSpec],
        fusion: LoadedFusion,
        threshold: float = 0.5,
        device: str = "cuda",
    ) -> None:
        unknown = sorted(set(streams) - set(fusion.stream_dims))
        if unknown:
            raise ValueError(
                f"Streams the fusion head was not trained on: {', '.join(unknown)}"
            )
        self.preprocessor = preprocessor
        self.streams = streams
        self.fusion = fusion
        self.threshold = threshold
        self.device = device

    def prepare(self, path: Path, kind: str):
        """The views this media kind can produce, by the matching entry point."""
        record = _Record(clip_id=path.stem)
        if kind == media_kind.VIDEO:
            return self.preprocessor.prepare(record, path)
        if kind == media_kind.IMAGE:
            return self.preprocessor.prepare_image(record, path)
        if kind == media_kind.AUDIO:
            return self.preprocessor.prepare_audio(record, path)
        raise ValueError(f"Cannot prepare media of kind {kind!r}")

    def predict(self, path: Path) -> PredictionResult:
        kind = media_kind.classify(path)
        if kind == media_kind.UNKNOWN:
            return PredictionResult(
                clip_id=Path(path).stem,
                verdict="indeterminate",
                probability=None,
                branch_logits={},
                blockers=("unsupported_media_type",),
                preprocessing_fingerprint="",
            )

        prepared = self.prepare(Path(path), kind)
        # Ask the routing table, not the views. A stream whose views happen to
        # exist but which this media kind should not drive must not run: an
        # image produces a visual_view, and the emotion stream reads that same
        # view, so without this an image would silently reach a model that also
        # expects a voice.
        allowed = set(media_kind.runnable(kind))

        logits: dict[str, float] = {}
        embeddings: dict[str, tuple[float, ...]] = {}
        blockers: list[str] = list(prepared.quality.full_fusion_blockers())

        with torch.inference_mode():
            for name, spec in self.streams.items():
                if _modality(name) not in allowed:
                    continue
                try:
                    logit, embedding = _forward(spec, prepared, self.device)
                except LookupError as missing:
                    blockers.append(f"missing_{missing.args[0]}")
                    continue
                if not np.isfinite(logit):
                    blockers.append(f"non_finite_{name}")
                    continue
                logits[name] = logit
                embeddings[name] = embedding

        if not embeddings:
            blockers.append("no_stream_could_read_this_clip")
            return PredictionResult(
                clip_id=prepared.clip_id,
                verdict="indeterminate",
                probability=None,
                branch_logits=logits,
                blockers=tuple(dict.fromkeys(blockers)),
                preprocessing_fingerprint=prepared.preprocessing_fingerprint,
            )

        probability = fuse(self.fusion, embeddings, self.device)
        return PredictionResult(
            clip_id=prepared.clip_id,
            verdict="fake" if probability >= self.threshold else "real",
            probability=probability,
            branch_logits=logits,
            # Quality blockers are reported even when a verdict is issued. A
            # face covering 60 percent of the frames still produces a number,
            # and the reader should see why it might be weak.
            blockers=tuple(dict.fromkeys(blockers)),
            preprocessing_fingerprint=prepared.preprocessing_fingerprint,
        )


def _modality(stream_name: str) -> str:
    """Which routing entry a stream belongs to, from its name.

    Matched by substring for the same reason the trainer does it: a checkpoint
    called `visual-dinov3` or `final-audio-seed17` should land in the right
    group without a registry of names to keep in step.
    """
    lowered = stream_name.lower()
    for modality in ("lipsync", "emotion", "audio"):
        if modality in lowered:
            return modality
    return "visual"
