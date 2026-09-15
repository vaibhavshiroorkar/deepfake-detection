"""Run a trained cross-modal stream on one clip, stage by stage.

The older `cross_modal.py` renders the same two pages over *random* projections,
because when it was written neither stream existed. Both are trained now, so
this runs the real thing: the checkpoint, the real encoders, the real attention
weights, the real logit.

That matters beyond freshness. The attention map on this page is the evidence
for the project's central finding. `diagonal_mass`, the fraction of attention
mass near the diagonal, is recorded every epoch during training and never enters
the loss, which is what makes it evidence about the mechanism rather than
another thing the model fits. Measured on the trained checkpoints it sits at
chance in every epoch of three runs: emotion 0.3434 to 0.3457 against a chance
of 0.3438 at eight query steps, lip-sync 0.0592 to 0.0595 against 0.0592 at
fifty. The streams score well and are not measuring correspondence.

So this page shows a map that looks like an attention map and a number saying it
carries no more structure than a uniform one would. Both are printed, because
the picture alone reads as a finding whether or not it is one.

Preprocessing here is the pipeline's own, not the dashboard's lighter path, so
the tensors are exactly what the model was trained on and the logit printed here
is the logit the batch pipeline would record.
"""

from __future__ import annotations

import numpy as np

# Chance diagonal mass depends on the query length, and comparing against the
# wrong one is a mistake this project already made: emotion's 0.3439 was called
# six times chance using lip-sync's 50-step baseline. Band 1 covers 2*1+1
# positions per row.
BAND = 1


def chance_diagonal_mass(queries: int, keys: int, band: int = BAND) -> float:
    """What uniform attention scores, for this map's shape."""
    rows = np.arange(queries)[:, None]
    cols = np.arange(keys)[None, :]
    inside = (np.abs(rows * (keys / queries) - cols) <= band).sum()
    return float(inside / (queries * keys))


def measured_diagonal_mass(weights: np.ndarray, band: int = BAND) -> float:
    """Fraction of attention mass within `band` of the diagonal."""
    rows, cols = weights.shape
    r = np.arange(rows)[:, None]
    c = np.arange(cols)[None, :]
    return float(weights[np.abs(r * (cols / rows) - c) <= band].sum() / rows)


def prepare_clip(video_path: str, stream: str):
    """The pipeline's own views for this clip, as the trained model reads them.

    Uses `views.preprocessor`, not the dashboard's lighter decoding path, so the
    numbers on the page match what the batch pipeline records for the same file.
    """
    from pathlib import Path

    from deepfake_detection.dashboard.configuration import dashboard_defaults
    from deepfake_detection.data.manifest import ClipRecord
    from deepfake_detection.inference.loading import build_preprocessor

    defaults = dashboard_defaults(root=Path.cwd())
    preprocessor = build_preprocessor(
        code_version=defaults.stream_code_version, device="cuda"
    )
    path = Path(video_path)
    record = ClipRecord(
        clip_id=path.stem,
        dataset="dashboard",
        video_path=path,
        manipulation_type="RealVideo-RealAudio",
        method="unknown",
        source="dashboard",
        targets=(),
        clip_fake=False,
        video_fake=False,
        audio_fake=False,
    )
    return preprocessor.prepare(record, path)


def run_stream(stream: str, checkpoint, prepared, device: str = "cuda") -> dict:
    """One forward pass, keeping every intermediate the page draws."""
    import torch

    from deepfake_detection.data.datasets import (
        STREAM_FRAME_LIMITS,
        STREAM_VIEWS,
        _normalize_waveform,
    )
    from deepfake_detection.streams.cross_modal_stream import (
        build_emotion_stream,
        build_lipsync_stream,
    )

    builder = build_lipsync_stream if stream == "lipsync" else build_emotion_stream
    model = builder(pretrained=checkpoint is None, common_dim=256)
    report = None
    if checkpoint is not None:
        from deepfake_detection.dashboard.lib import checkpoints as checkpoints_lib

        report = checkpoints_lib.load_into(model, checkpoint)
    model = model.to(device).eval()

    video_field, audio_field = STREAM_VIEWS[stream]
    video = getattr(prepared, video_field)
    audio = getattr(prepared, audio_field)
    if video is None or audio is None:
        missing = video_field if video is None else audio_field
        return {"missing": missing, "report": report}

    limit = STREAM_FRAME_LIMITS[stream]
    if limit is not None:
        video = video[:limit]
    video_tensor = (
        torch.from_numpy(np.asarray(video, dtype=np.float32)).unsqueeze(0).to(device)
    )
    audio_tensor = _normalize_waveform(
        torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0).to(device)
    )
    with torch.inference_mode():
        output = model(video=video_tensor, audio=audio_tensor)

    weights = output.attention[0].float().cpu().numpy()
    return {
        "missing": None,
        "report": report,
        "logit": float(output.logit[0].cpu()),
        "probability": float(torch.sigmoid(output.logit[0]).cpu()),
        "embedding": output.embedding[0].float().cpu().numpy(),
        "attention": weights,
        "video": np.asarray(video, dtype=np.float32),
        "audio": np.asarray(audio, dtype=np.float32),
        "counts": model.param_counts() if hasattr(model, "param_counts") else {},
    }
