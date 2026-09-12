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
from deepfake_detection.fusion.deep_loading import LoadedFusion, fuse, load_fusion
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
        # A stream the head knows about and this server has no checkpoint for.
        # Not an error: the head masks an absent stream to exactly zero, which is
        # the same thing it does for a clip that could not be read. It is
        # reported on every verdict, because a fusion of three where five were
        # expected is a weaker claim and the reader cannot see it otherwise.
        self.unavailable = tuple(sorted(set(fusion.stream_dims) - set(streams)))
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
                media_kind=kind,
            )

        prepared = self.prepare(Path(path), kind)
        # Ask the routing table, not the views. A stream whose views happen to
        # exist but which this media kind should not drive must not run: an
        # image produces a visual_view, and the emotion stream reads that same
        # view, so without this an image would silently reach a model that also
        # expects a voice.
        allowed = set(media_kind.runnable(kind))
        absent = tuple(
            name for name in self.unavailable if _modality(name) in allowed
        )

        logits: dict[str, float] = {}
        embeddings: dict[str, tuple[float, ...]] = {}
        blockers: list[str] = list(prepared.quality.full_fusion_blockers())
        blockers.extend(f"no_checkpoint_for_{name}" for name in absent)

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
                media_kind=kind,
            )

        probability = fuse(self.fusion, embeddings, self.device)
        # The head carries a cut-off per media kind because an image verdict and
        # a video verdict come from different evidence through the same weights.
        # A kind the head was never calibrated for falls back to the engine
        # default and says so, rather than being read as if it were calibrated.
        threshold = self.fusion.threshold_for(kind)
        if threshold is None:
            threshold = self.threshold
            blockers.append(f"uncalibrated_threshold_for_{kind}")
        return PredictionResult(
            clip_id=prepared.clip_id,
            verdict="fake" if probability >= threshold else "real",
            probability=probability,
            branch_logits=logits,
            # Quality blockers are reported even when a verdict is issued. A
            # face covering 60 percent of the frames still produces a number,
            # and the reader should see why it might be weak.
            blockers=tuple(dict.fromkeys(blockers)),
            preprocessing_fingerprint=prepared.preprocessing_fingerprint,
            media_kind=kind,
            threshold=threshold,
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


def load_multimodal_engine(
    *,
    run_dir: Path,
    code_version: str,
    fusion_path: Path | None = None,
    device: str = "cuda",
    threshold: float = 0.5,
    root: Path | None = None,
) -> MultimodalEngine:
    """Build the served engine from one run directory.

    The preprocessing hash is checked against the head's, not assumed. A server
    whose view settings differ from the cache the streams were trained on
    produces face crops of a different size or a waveform with a different
    silence rule, and every number that follows is quietly wrong. The two hashes
    are the only thing that catches it.
    """
    from deepfake_detection.fusion.stream_loading import load_streams
    from deepfake_detection.inference.loading import build_preprocessor
    from deepfake_detection.views.cache import preprocessing_config_hash

    fusion = load_fusion(
        fusion_path or run_dir / "checkpoints" / "gate-fusion.pt", device=device
    )
    preprocessor = build_preprocessor(code_version=code_version, device=device)
    actual = preprocessing_config_hash(
        config=preprocessor.config, code_version=code_version
    )
    if fusion.preprocessing_hash and actual != fusion.preprocessing_hash:
        raise ValueError(
            "This server's preprocessing does not match the fusion head: it "
            f"computes {actual[:12]} and the head was fitted on features built "
            f"with {fusion.preprocessing_hash[:12]}. Serving through it would "
            "feed the streams inputs they were not trained on."
        )

    specs = load_streams(
        run_dir, device, root=root, only=tuple(fusion.stream_dims)
    )
    if not specs:
        raise ValueError(
            f"None of the head's streams have checkpoints in {run_dir}: "
            f"{', '.join(sorted(fusion.stream_dims))}."
        )
    return MultimodalEngine(
        preprocessor=preprocessor,
        streams={spec.name: spec for spec in specs},
        fusion=fusion,
        threshold=threshold,
        device=device,
    )
