"""One visual stream, one clip, every stage in between.

The Preprocessing page walks a clip to the tensor a model receives. This page
picks it up there and keeps going: fold, backbone stages, per-frame features,
temporal model, projection, head. Every picture on it is measured from a real
forward pass over the selected clip, captured by
models/streams/common/introspect.py.

Nothing runs until Run is clicked. A trace is a full forward pass over sixteen
224-pixel frames, which is not something to do on every slider nudge, so the
result is parked in session_state and the page redraws from it. Changing the
architecture invalidates it, and the page says so instead of showing stale
pictures beside new settings.
"""
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import streamlit as st

from dashboard.lib import stream_pages, stream_ui, trace_ui
from models.streams.common import introspect

TRACE_KEY = "visual_trace"
SIG_KEY = "visual_trace_signature"
TOP_CHANNELS = 8

st.title("Visual stream")
st.caption("Artifact-focused backbones over the face-crop sequence, never audio. All three are "
           "the same module with a different backbone name, so what changes between them is "
           "exactly what you see below.")

# --------------------------------------------------------------- what to run

names = {key: name for key, (name, _) in stream_pages.VISUAL_MODELS.items()}
keys = list(names)
default = st.session_state.get("visual_backbone", keys[0])
chosen_name = st.segmented_control(
    "Backbone", [names[k] for k in keys], default=names.get(default, names[keys[0]]),
    key="visual_backbone_pick")
key = next((k for k in keys if names[k] == chosen_name), keys[0])
st.session_state["visual_backbone"] = key
backbone_name = stream_pages.VISUAL_MODELS[key][1]

with st.container(border=True):
    st.markdown("**Architecture**")
    st.caption(f"`{backbone_name}`  ·  shared with the Streams hub, so a change here follows you "
               "back there.")
    stream_pages.render_config_controls(st, key, ns="visual")

with st.container(border=True):
    st.markdown("**Weights**")
    checkpoint = stream_ui.render_checkpoint_picker(st, key, ns=f"visual_{key}")

with st.container(border=True):
    st.markdown("**Run**")
    video_path = stream_pages.render_inherited_clip(st)
    config = stream_pages.build_config(key)
    # The frame count follows the Preprocessing page's slider, so lowering it
    # there can strand this value above its own maximum, which Streamlit treats
    # as an error rather than clamping.
    last_frame = config.num_frames - 1
    if st.session_state.get("visual_detail", 0) > last_frame:
        st.session_state["visual_detail"] = last_frame
    c1, c2, c3 = st.columns([2, 2, 3])
    # Streamlit warns when a widget is given both a default and a session_state
    # value, so the default is only offered on the first run.
    first_run = {} if "visual_detail" in st.session_state else {"value": 0}
    detail_frame = c1.number_input(
        "Detail frame", 0, last_frame, key="visual_detail", **first_run,
        help="Which frame keeps its full activations. Every frame gets a summary map; "
             "holding all channels of all frames would cost a few hundred megabytes.")
    fixed_seed = c2.checkbox(
        "Fixed seed", value=True, key="visual_fixed_seed",
        help="Seed before building, so an untrained model gives the same answer twice. "
             "Ignored once a checkpoint is loaded.")
    seed = c2.number_input("Seed", 0, 999999, 42, key="visual_seed", disabled=not fixed_seed)
    run = c3.button("Run this clip through the model", type="primary", width="stretch",
                    disabled=video_path is None, key="visual_run")

signature = (key, config.temporal_type, config.temporal_hidden, config.common_dim,
             config.freeze_backbone, str(checkpoint), video_path, int(detail_frame),
             bool(fixed_seed), int(seed))


def _run_trace():
    import torch
    from dashboard.lib import inference, media
    from models.streams.common.visual_stream import build_visual_stream

    if fixed_seed:
        torch.manual_seed(int(seed))
    with st.spinner("Building the model…"):
        model = build_visual_stream(config).eval()

    report = None
    if checkpoint is not None:
        with st.spinner("Loading checkpoint…"):
            from dashboard.lib import checkpoints
            report = checkpoints.load_into(model, checkpoint)

    detector, _device = media.get_detector()
    with st.spinner("Detecting faces…"):
        faces = inference.decode_face_clip(str(video_path), config.num_frames, detector)
        frames = inference.frames_to_tensor(faces)
    with st.spinner("Forward pass…"):
        trace = introspect.trace_visual_stream(model, frames, int(detail_frame))
    return {"trace": trace, "counts": model.param_counts(), "report": report,
            "crops": faces, "input": frames.squeeze(0).numpy()}


if run:
    st.session_state[TRACE_KEY] = _run_trace()
    st.session_state[SIG_KEY] = signature

result = st.session_state.get(TRACE_KEY)
if result is None:
    st.info("Pick a clip on the **Preprocessing** page, then run it here. Every picture below "
            "is measured from that forward pass, so there is nothing to show until one happens.")
    st.stop()
if st.session_state.get(SIG_KEY) != signature:
    st.warning("These settings have changed since the trace below was captured. Run again to "
               "see the model you have configured now.")

trace = result["trace"]
counts = result["counts"]
if result["report"] is not None:
    stream_ui.report_load(st, result["report"])

m1, m2, m3, m4 = st.columns(4)
m1.metric("Backbone features", counts["feature_dim"])
m2.metric("Embedding dim", counts["embedding_dim"])
m3.metric("Total params", f"{counts['total'] / 1e6:.1f}M")
m4.metric("Trainable params", f"{counts['trainable'] / 1e6:.1f}M")

st.divider()

# ------------------------------------------------------------------ 0 · input

with st.container(border=True):
    left, right = st.columns([1, 2])
    left.markdown("**0 · Input**")
    left.caption(f"The face tensor the Preprocessing page ends on: `{trace.input_shape}`. "
                 "ImageNet-normalised, shown de-normalized.")
    left.metric("Frames", config.num_frames)
    trace_ui.show_frames(right, trace_ui.denormalize(result["input"]),
                         "Face crops, in clip order")

# ------------------------------------------------------------------- 1 · fold

with st.container(border=True):
    left, right = st.columns([1, 2])
    left.markdown("**1 · Fold**")
    left.caption("Frames move into the batch dimension so the backbone sees a flat stack of "
                 "images. It has no notion of time; that is the temporal model's job at step 4.")
    right.code(f"{trace.input_shape}   ->   {trace.folded_shape}\n"
               f"[B, T, 3, H, W]      ->   [B*T, 3, H, W]", language="text")

# --------------------------------------------------------------- 2 · backbone

st.subheader("2 · Backbone stages")
is_vit = trace.stages[0].kind == introspect.TOKENS
if is_vit:
    st.caption("DINOv2 is a transformer, so there are no channel maps to show. Each block emits "
               "one token per 14-pixel patch plus a CLS token, and the maps below are measured "
               "from those tokens: how much weight each patch's token carries, and how closely "
               "it points in the same direction as the CLS token that summarises the frame.")
else:
    st.caption("Each stage's map is the mean over its channels, so it shows where the stage "
               "responded rather than to what. Maps are upsampled without interpolation: a 7x7 "
               "stage really does see the face as 49 regions, and smoothing that would imply a "
               "precision the network does not have.")

crops = result["crops"]
detail = trace.detail_frame

if is_vit:
    labels = [s.spec.name for s in trace.stages]
    shown = st.multiselect("Blocks to show", labels, default=[labels[0], labels[len(labels) // 2],
                                                              labels[-1]],
                           key="visual_vit_blocks",
                           help="Twelve blocks all at once is a wall; these three show the "
                                "beginning, middle and end of the stack.")
    stages = [s for s in trace.stages if s.spec.name in shown]
else:
    stages = trace.stages

for stage in stages:
    with st.container(border=True):
        left, right = st.columns([1, 2])
        with left:
            st.markdown(f"**{stage.spec.name}**")
            st.caption(f"shape `{stage.shape}` per frame")
            if stage.spec.reduction:
                st.caption(f"{stage.spec.channels} channels  ·  1/{stage.spec.reduction} "
                           "of the input resolution")
            st.image(trace_ui.overlay(crops[detail], stage.summary[detail]),
                     caption=f"frame {detail}, over its crop", width="stretch")
        with right:
            trace_ui.show_frames(
                right, [trace_ui.heatmap(m, size=112) for m in stage.summary],
                "Response across the clip"
                + ("  ·  patch-token norm" if is_vit else "  ·  mean over channels"))
            if is_vit:
                g1, g2 = st.columns(2)
                g1.image(trace_ui.heatmap(introspect.patch_token_map(stage.detail)),
                         caption=f"token norm, frame {detail}", width="stretch")
                g2.image(trace_ui.heatmap(introspect.cls_similarity_map(stage.detail)),
                         caption=f"cosine to CLS, frame {detail}", width="stretch")
            else:
                top = introspect.top_channels(stage.detail, TOP_CHANNELS)
                trace_ui.show_frames(
                    right, trace_ui.channel_grid(stage.detail, top),
                    f"The {TOP_CHANNELS} strongest channels of frame {detail}: "
                    f"#{', #'.join(str(i) for i in top)}")

# ------------------------------------------------------- 3 · per-frame features

with st.container(border=True):
    left, right = st.columns([1, 2])
    feats = trace.frame_features
    left.markdown("**3 · Pool and unfold**")
    left.caption(f"Global average pooling collapses each frame's final map to one vector, and "
                 f"the stack is folded back into a sequence: `{feats.shape}`.")
    norms = np.linalg.norm(feats, axis=1)
    left.metric("Feature dim", feats.shape[1])
    left.metric("Frame-to-frame spread", f"{norms.std() / max(norms.mean(), 1e-9):.3f}",
                help="Standard deviation of the per-frame feature norm over its mean. A forgery "
                     "that flickers between frames has more to vary here than a stable one, "
                     "though an untrained backbone's number means nothing.")
    right.pyplot(trace_ui.matrix_fig(feats.T, f"Per-frame features {feats.shape}",
                                     "frame", "feature"))
    right.pyplot(trace_ui.line_fig(norms, "Feature norm per frame"))

# --------------------------------------------------------------- 4 · temporal

with st.container(border=True):
    left, right = st.columns([1, 2])
    left.markdown("**4 · Temporal model**")
    label = stream_pages.settings(key)["temporal"]
    if trace.temporal_sequence is None:
        left.caption("Mean-pool: the frame vectors are averaged. Ordering is discarded entirely, "
                     "which is the point of comparing it against the recurrent options.")
        left.metric("Mode", label)
        right.pyplot(trace_ui.line_fig(trace.clip_vector[:256],
                                       "Clip vector (first 256 dimensions of the mean)",
                                       xlabel="dimension"))
    else:
        seq = trace.temporal_sequence
        left.caption(f"{label} reads the sequence and keeps a running state. The clip vector is "
                     "its final state, so frame 16 is not weighted more than frame 1: a "
                     "bidirectional pass has read the sequence from both ends.")
        left.metric("Mode", label)
        left.metric("Output per step", seq.shape[1])
        right.pyplot(trace_ui.matrix_fig(seq.T, f"Hidden state per frame {seq.shape}",
                                         "frame", "unit"))
        right.pyplot(trace_ui.vector_fig(trace.clip_vector,
                                         f"Final state -> clip vector {trace.clip_vector.shape}"))

# ------------------------------------------------------------- 5 · projection

with st.container(border=True):
    left, right = st.columns([1, 2])
    left.markdown("**5 · Projection**")
    left.caption("Linear then LayerNorm, down to the shared width. This vector is the stream's "
                 "actual product: it is what the feature store keeps and what fusion reads.")
    left.metric("Embedding dim", trace.embedding.shape[0])
    left.metric("L2 norm", f"{np.linalg.norm(trace.embedding):.2f}")
    right.pyplot(trace_ui.vector_fig(trace.embedding,
                                     f"Clip embedding {trace.embedding.shape}"))

# --------------------------------------------------------------- 6 · dev head

with st.container(border=True):
    left, right = st.columns([1, 2])
    left.markdown("**6 · Development head**")
    left.caption("A single linear layer to one logit, so a stream's standalone power can be "
                 "measured before fusion exists. Fusion discards it and reads the embedding.")
    right.metric("Fake probability", f"{trace.prob:.3f}")
    right.caption(f"logit {trace.logit:.4f}")
    if checkpoint is None:
        right.warning("These weights are untrained, so this number is a plumbing check and not a "
                      "detection. Load a checkpoint to make it mean something.")

st.divider()
st.subheader("Shape ladder")
stage_lines = "\n".join(f"  {s.spec.name:<16} {str(s.shape):<20} per frame" for s in trace.stages)
st.code(
    f"input     {trace.input_shape}\n"
    f"fold      {trace.folded_shape}\n"
    f"{stage_lines}\n"
    f"features  {trace.frame_features.shape}\n"
    f"temporal  {'(mean-pool)' if trace.temporal_sequence is None else trace.temporal_sequence.shape}\n"
    f"clip vec  {trace.clip_vector.shape}\n"
    f"embedding {trace.embedding.shape}   -> feature store -> fusion\n"
    f"logit     {trace.logit:.4f}   (dev head, discarded at fusion)",
    language="text")
