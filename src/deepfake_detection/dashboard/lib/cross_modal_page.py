"""The Lip-Sync and Emotion pages, running the trained streams stage by stage.

Same shape as the Visual stream page: pick a clip, pick weights, press Run, and
every picture below comes from that forward pass. One renderer serves both
streams, differing only by which pair of views it reads.

The order of the stages is the order the tensors move through the model, so the
page reads as the model runs: inputs, encoders, cross-attention, the validity
check, then the score.
"""

from __future__ import annotations

import numpy as np

STREAMS = {
    "lipsync": {
        "title": "Lip-sync stream",
        "caption": (
            "Mouth crops against the waveform. Cross-attention asks how cleanly "
            "the sound answers the articulation."
        ),
        "video_label": "Mouth crops",
        "checkpoint_stream": "stream-lipsync",
        "query": "audio",
    },
    "emotion": {
        "title": "Emotion stream",
        "caption": (
            "Face crops against the waveform. Cross-attention asks how cleanly "
            "vocal affect answers facial affect."
        ),
        "video_label": "Face crops",
        "checkpoint_stream": "stream-emotion",
        "query": "audio",
    },
}


def _frames_for_display(view: np.ndarray, limit: int = 8) -> list[np.ndarray]:
    """De-normalise the cached crops back to something showable."""
    from deepfake_detection.preprocessing.ops.constants import (
        IMAGENET_MEAN,
        IMAGENET_STD,
    )

    values = np.asarray(view, dtype=np.float32)[:limit]
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(1, 3, 1, 1)
    pixels = np.clip((values * std + mean) * 255.0, 0, 255)
    return list(pixels.transpose(0, 2, 3, 1).astype(np.uint8))


def render(st, stream: str) -> None:
    from deepfake_detection.dashboard.lib import (
        cross_modal_live,
        sticky,
        stream_pages,
        stream_ui,
        trace_ui,
    )

    spec = STREAMS[stream]
    st.title(spec["title"])
    st.caption(spec["caption"])

    video_path = stream_pages.render_inherited_clip(st)
    run_store = sticky.run_state(f"cm_{stream}")

    with st.container(border=True):
        st.markdown("**Weights**")
        checkpoint = stream_ui.render_checkpoint_picker(
            st, spec["checkpoint_stream"], ns=f"cm_{stream}", store=run_store
        )
        run = st.button(
            "Run this clip through the model",
            type="primary",
            width="stretch",
            disabled=video_path is None,
            key=f"cm_run_{stream}",
        )

    if video_path is None:
        st.info("Choose a clip on the Preprocessing page or in the Streams hub.")
        return

    signature = (stream, str(checkpoint), str(video_path))
    if run:
        with st.spinner("Preprocessing the clip the way the pipeline does…"):
            prepared = cross_modal_live.prepare_clip(str(video_path), stream)
        with st.spinner("Forward pass…"):
            run_store["result"] = cross_modal_live.run_stream(
                stream, checkpoint, prepared
            )
            run_store["signature"] = signature
            run_store["quality"] = {
                "face_coverage": prepared.quality.face_coverage,
                "blockers": list(prepared.quality.full_fusion_blockers()),
            }

    result = run_store.get("result")
    if result is None:
        st.info("Press Run to trace this clip through the stream.")
        return
    if run_store.get("signature") != signature:
        st.warning(
            "The clip or the weights changed since this run. Press Run again.",
            icon=":material/refresh:",
        )

    if result.get("missing"):
        st.error(
            f"This clip has no `{result['missing']}`, so the stream cannot read "
            "it. That is the abstention policy working: a clip with no stable "
            "face track or no audio is reported rather than scored."
        )
        return

    report = result.get("report")
    if checkpoint is None:
        st.warning(
            "No checkpoint selected, so the encoders are pretrained but the "
            "cross-attention and the head are random. Everything below is real "
            "arithmetic over an untrained model, and the score means nothing.",
            icon=":material/warning:",
        )
    elif report is not None:
        st.caption(f"Loaded `{getattr(checkpoint, 'name', checkpoint)}`")

    # -------------------------------------------------------------- verdict
    score = result["probability"]
    calls_fake = score >= 0.5
    with st.container(border=True):
        v1, v2 = st.columns([1, 2])
        v1.metric(
            "Score, 0 to 1",
            f"{score:.3f}",
            "fake" if calls_fake else "real",
            delta_color="inverse" if calls_fake else "normal",
        )
        v2.markdown(f"### This stream calls it **{'FAKE' if calls_fake else 'REAL'}**")
        v2.caption(
            f"Score {score:.3f} against a 0.500 cut-off, from the "
            f"{spec['title'].lower()} alone. 0 is authentic, 1 is manipulated."
        )
        if checkpoint is None:
            v2.caption(
                "No checkpoint loaded, so this number came from an untrained "
                "head and means nothing."
            )
    st.caption(
        "One stream is not the system's verdict. The Evidence gate fuses every "
        "stream the media kind can drive and applies a threshold chosen per "
        "kind, which for video is 0.50 but for an image is 0.76. This page "
        "exists to show what one stream saw and why."
    )

    # ------------------------------------------------------------ 1 · inputs
    st.subheader("1 · What the two encoders receive")
    left, right = st.columns([3, 2])
    frames = _frames_for_display(result["video"])
    left.image(
        frames,
        width=96,
        caption=[f"{i}" for i in range(len(frames))],
    )
    left.caption(
        f"{spec['video_label']}, {result['video'].shape[0]} frames at "
        f"{result['video'].shape[-1]} pixels. These are the cached tensors, "
        "de-normalised for display only."
    )
    audio = result["audio"]
    right.line_chart(audio[:: max(1, len(audio) // 2000)], height=140)
    right.caption(
        f"Waveform, {len(audio):,} samples. Normalised the same way the dataset "
        "normalises it, since a stream trained on one normalisation and scored "
        "on another reads a different input and nothing fails loudly."
    )

    # ---------------------------------------------------------- 2 · attention
    st.subheader("2 · Cross-attention")
    weights = result["attention"]
    queries, keys = weights.shape
    st.caption(
        f"softmax(QK^T / sqrt(d)) V over a [{queries}, {keys}] map. Audio asks "
        "the questions; the visual sequence supplies the evidence."
    )
    c1, c2 = st.columns([3, 2])
    c1.image(
        trace_ui.heatmap(weights),
        width="stretch",
        caption="Attention weights. Rows are audio steps, columns visual steps.",
    )
    measured = cross_modal_live.measured_diagonal_mass(weights)
    chance = cross_modal_live.chance_diagonal_mass(queries, keys)
    c2.metric("Diagonal mass", f"{measured:.4f}", f"{measured - chance:+.4f} vs chance")
    c2.caption(
        f"Chance for a [{queries}, {keys}] map is {chance:.4f}. The baseline "
        "depends on the query length, and comparing against the wrong one is a "
        "mistake this project made once: emotion's 0.3439 was called six times "
        "chance using lip-sync's fifty-step baseline."
    )

    # ------------------------------------------------------- 3 · the check
    st.subheader("3 · Does the mechanism operate?")
    ratio = measured / chance if chance else float("nan")
    if measured > chance * 1.15:
        st.success(
            f"Diagonal mass is {ratio:.2f} times chance. Attention is "
            "concentrating near the diagonal, which is what a stream reading "
            "correspondence should do."
        )
    else:
        st.warning(
            f"Diagonal mass is {ratio:.2f} times chance, so the attention "
            "carries no more diagonal structure than a uniform map would. "
            "Whatever this stream is reading, it is not the alignment between "
            "the two modalities.",
            icon=":material/science:",
        )
        st.caption(
            "This is the project's central finding rather than a bug on this "
            "page. Measured across every epoch of three training runs: emotion "
            "0.3434 to 0.3457 against a chance of 0.3438, lip-sync 0.0592 to "
            "0.0595 against 0.0592. The emotion stream reaches 0.9978 in-domain "
            "and 0.9969 on LAV-DF, whose matched windows come from one file so "
            "correspondence is the only cue the corpus leaves, with its "
            "attention still at chance. `diagonal_mass` is recorded every epoch "
            "and never enters the loss, which is what makes it evidence."
        )

    # ------------------------------------------------------------- 4 · score
    st.subheader("4 · How that score was produced")
    s1, s2, s3 = st.columns(3)
    s1.metric("Logit", f"{result['logit']:+.3f}")
    s2.metric("Score after sigmoid", f"{score:.3f}")
    s3.metric("Embedding width", f"{result['embedding'].shape[0]}")
    st.caption(
        "The head emits a logit; the sigmoid maps it to 0 to 1. The embedding "
        "beside it is what the fusion head consumes, and it carries more than "
        "the single number does, which is why fusion is not just an average of "
        "stream scores."
    )
    st.caption(
        "Treat the score as uncalibrated off-corpus. Expected calibration error "
        "measured 0.0022 in-domain and 0.6366 cross-corpus, so a confident "
        "number on unfamiliar footage is not evidence of confidence."
    )
    quality = run_store.get("quality") or {}
    if quality.get("blockers"):
        st.caption("Quality blockers on this clip: " + ", ".join(quality["blockers"]))
