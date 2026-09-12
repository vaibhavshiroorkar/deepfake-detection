"""Shared state and controls for the Streams section.

The section is a hub plus three subpages. The hub (the dashboard's pages/streams.py)
gives quick control over all three streams at once; each subpage takes one stream
and walks a clip through it step by step. Both need the same architecture
controls and the same idea of which streams are enabled, so both come from here.

Configuration is stored in plain session_state dicts (`stream_cfg_<key>`) rather
than read off the widgets. Streamlit discards widget state for widgets that were
not rendered on the current run, so a hub setting would reset itself the moment
you navigated to a subpage and back. The dicts survive; the widgets initialise
from them and write back.

the dashboard's lib/sticky.py applies the same rule to the other two things that cross
pages: the Preprocessing page's frame count and audio window, and each backbone's
last run. Anything read on a page other than the one whose widget wrote it
belongs in one of these stores.
"""

import streamlit as st

from deepfake_detection.dashboard.lib import sticky
from deepfake_detection.streams.config import (
    DINOV3,
    EFFICIENTNET_B0,
    StreamConfig,
)

# Label -> (temporal_type, bidirectional), the three ways to collapse a frame
# sequence into one clip vector.
# The unidirectional pair is here because the trained checkpoints need it:
# `ddf train visual` builds a unidirectional GRU, so without these two entries
# the picker can tell you the file wants "GRU (unidirectional)" and the control
# has no such setting, leaving the temporal model random no matter what you
# choose.
TEMPORAL = {
    "BiLSTM": ("lstm", True),
    "GRU": ("gru", True),
    "LSTM (unidirectional)": ("lstm", False),
    "GRU (unidirectional)": ("gru", False),
    "Mean-pool": ("mean", False),
}

# The three visual backbones, in the order the Documentation page introduces
# them. Each is the same module with a different backbone name.
VISUAL_MODELS = {
    "efficientnet": ("EfficientNet-B0", EFFICIENTNET_B0),
    "dinov3": ("DINOv3 (ViT-S/16)", DINOV3),
}

DEFAULTS = {
    "enabled": True,
    "temporal": "BiLSTM",
    "hidden": 256,
    "dim": 256,
    "freeze": True,
}

# How each backbone reduces its per-frame output to one vector. Not a tuning
# choice and not exposed as a control: DINOv3 must pool over the CLS token, and
# the trained checkpoints were written under these values. Building a stream
# with the wrong one still loads every tensor -- the shapes match either way --
# and then quietly computes a different vector than the one that was trained.
GLOBAL_POOL = {"efficientnet": "avg", "dinov3": "token"}

# Only the visual streams are configurable: the cross-modal encoders are Stage
# 4 and 5, so there is nothing yet to configure for them.
CROSS_MODAL = {
    "lipsync": ("Lip-Sync", "AV-HuBERT + Whisper", 4),
    "emotion": ("Emotion", "HSEmotions + Wav2Vec2", 5),
}


def settings(key: str) -> dict:
    """The stored architecture settings for one model, created on first use."""
    return st.session_state.setdefault(f"stream_cfg_{key}", dict(DEFAULTS))


def build_config(key: str) -> StreamConfig:
    """A StreamConfig from the stored settings, ready to build.

    pretrained=False because no weights are downloaded here; a trained checkpoint
    is loaded afterwards when one exists. grad_checkpointing off because it only
    saves memory during a backward pass, and there is never one in this app.

    num_frames follows the Preprocessing page's slider rather than the config
    default, so the sequence a stream reads here is the sequence that page just
    showed you. The batch pipeline fixes it at 16.
    """
    current = settings(key)
    temporal_type, bidirectional = TEMPORAL[current["temporal"]]
    return StreamConfig(
        stream_name=key,
        backbone_name=VISUAL_MODELS[key][1],
        pretrained=False,
        temporal_type=temporal_type,
        temporal_bidirectional=bidirectional,
        temporal_hidden=int(current["hidden"]),
        common_dim=int(current["dim"]),
        freeze_backbone=bool(current["freeze"]),
        grad_checkpointing=False,
        global_pool=GLOBAL_POOL[key],
        frame_chunk_size=0,
        num_frames=int(sticky.clip_settings()["n_frames"]),
    )


# Every namespace that renders the architecture controls. Adopting a checkpoint
# has to clear the widget state in all of them: a Streamlit widget with an
# explicit key reads session_state and ignores the index it was given, so
# writing the settings dict alone leaves the control showing its old value.
CONTROL_NAMESPACES = ("visual", "hub")

# The reverse of TEMPORAL: from what a checkpoint's tensors say back to the
# label the control offers.
_TEMPORAL_LABELS = {value: label for label, value in TEMPORAL.items()}


def adopt_architecture(key: str, architecture: dict) -> bool:
    """Set the controls to the architecture a checkpoint was trained with.

    Returns whether anything changed, so the caller can rerun the page rather
    than render controls that disagree with the weights just loaded.

    Loading is non-strict by design, so a mismatch is reported and not raised.
    That makes it quiet: the backbone lands, the temporal model silently stays
    random, and the page shows a confident number computed from half a model.
    """
    label = _TEMPORAL_LABELS.get(
        (architecture.get("temporal"), bool(architecture.get("bidirectional")))
    )
    if label is None:
        return False

    current = settings(key)
    wanted = {"temporal": label}
    if architecture.get("hidden"):
        wanted["hidden"] = int(architecture["hidden"])
    if architecture.get("common_dim"):
        wanted["dim"] = int(architecture["common_dim"])
    if all(current.get(name) == value for name, value in wanted.items()):
        return False

    current.update(wanted)
    for namespace in CONTROL_NAMESPACES:
        for field in ("temporal", "hidden", "dim"):
            st.session_state.pop(f"{namespace}_{key}_{field}", None)
    return True


def render_config_controls(container, key: str, ns: str) -> dict:
    """The four architecture controls, writing back into the stored settings.

    `ns` namespaces the widget keys, so the hub and a subpage can both render the
    controls for the same model without colliding.
    """
    current = settings(key)
    labels = list(TEMPORAL)
    c1, c2 = container.columns(2)
    temporal = c1.selectbox(
        "Temporal model",
        labels,
        key=f"{ns}_{key}_temporal",
        index=labels.index(current["temporal"]),
    )
    hidden = c1.slider(
        "Temporal hidden",
        64,
        512,
        int(current["hidden"]),
        step=64,
        key=f"{ns}_{key}_hidden",
        disabled=temporal == "Mean-pool",
        help="Ignored when mean-pooling, which has no hidden state.",
    )
    dim = c2.select_slider(
        "Embedding dim",
        [128, 256, 512],
        int(current["dim"]),
        key=f"{ns}_{key}_dim",
        help="The width every stream is projected to before fusion.",
    )
    freeze = c2.checkbox(
        "Freeze backbone", value=bool(current["freeze"]), key=f"{ns}_{key}_freeze"
    )
    current.update(temporal=temporal, hidden=hidden, dim=dim, freeze=freeze)
    return current


def enabled_streams() -> list[str]:
    """Keys of the visual streams currently marked for inclusion in fusion."""
    return [key for key in VISUAL_MODELS if settings(key)["enabled"]]


def inherited_clip():
    """(clip_id, absolute path) of the clip chosen on the Preprocessing page, or None.

    Running a stream means one forward pass over one clip, and the clip you want
    is invariably the one you were just inspecting. Reading the Preprocessing
    page's selection instead of rendering a second picker also means an uploaded
    video flows straight through, and there is only one place a clip is chosen.
    """
    row = st.session_state.get("pp_row")
    path = st.session_state.get("pp_video_path")
    if not row or not path:
        return None
    return row.get("clip_id", "clip"), str(path)


def render_inherited_clip(container) -> str | None:
    """The clip this page runs on, and the control to change it.

    Three sources: the clip chosen on Preprocessing, a dataset clip, or an
    upload. Preprocessing is offered first when one exists, because the clip you
    want to run is invariably the one you were just inspecting.

    The page keeps its own selection rather than writing back into the
    Preprocessing page's slot. Sharing one slot would make "From Preprocessing"
    mean "whatever I last picked here", which is not a source.
    """
    chosen = st.session_state.get("stream_row")
    path = st.session_state.get("stream_video_path")
    if chosen and path:
        container.caption(f"Clip: **{chosen.get('clip_id', 'clip')}**")
        with container.expander("Choose a different clip"):
            _render_picker(container)
        return str(path)

    inherited = inherited_clip()
    if inherited is not None:
        clip_id, inherited_path = inherited
        container.caption(f"Clip: **{clip_id}**, from the Preprocessing page.")
        with container.expander("Choose a different clip"):
            _render_picker(container)
        return inherited_path

    container.caption("Pick a clip to run.")
    with container.container(border=True):
        _render_picker(container)
    return st.session_state.get("stream_video_path")


def _render_picker(container) -> None:
    """The three-source clip control, storing the result for this page."""
    from deepfake_detection.dashboard.lib import selectors

    row = selectors.render_selection(allow_preprocessing=True, key="stream_sel_source")
    if row is None:
        return
    video_path = selectors.clip_path(row)
    if not video_path.exists():
        container.error(f"Video not found: {video_path}")
        return
    st.session_state["stream_row"] = row.to_dict()
    st.session_state["stream_video_path"] = str(video_path)
