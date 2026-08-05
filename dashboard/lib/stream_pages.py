"""Shared state and controls for the Streams section.

The section is a hub plus three subpages. The hub (dashboard/pages/streams.py)
gives quick control over all three streams at once; each subpage takes one stream
and walks a clip through it step by step. Both need the same architecture
controls and the same idea of which streams are enabled, so both come from here.

Configuration is stored in plain session_state dicts (`stream_cfg_<key>`) rather
than read off the widgets. Streamlit discards widget state for widgets that were
not rendered on the current run, so a hub setting would reset itself the moment
you navigated to a subpage and back. The dicts survive; the widgets initialise
from them and write back.

dashboard/lib/sticky.py applies the same rule to the other two things that cross
pages: the Preprocessing page's frame count and audio window, and each backbone's
last run. Anything read on a page other than the one whose widget wrote it
belongs in one of these stores.
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

from dashboard.lib import sticky
from models.streams.common.config import (
    StreamConfig, DINOV3, EFFICIENTNET_B0, XCEPTION,
)

# Label -> (temporal_type, bidirectional), the three ways to collapse a frame
# sequence into one clip vector.
TEMPORAL = {"BiLSTM": ("lstm", True), "GRU": ("gru", True), "Mean-pool": ("mean", False)}

# The three visual backbones, in the order the Documentation page introduces
# them. Each is the same module with a different backbone name.
VISUAL_MODELS = {
    "xception": ("Xception", XCEPTION),
    "efficientnet": ("EfficientNet-B0", EFFICIENTNET_B0),
    "dinov3": ("DINOv3 (ViT-S/16)", DINOV3),
}

DEFAULTS = {"enabled": True, "temporal": "BiLSTM", "hidden": 256, "dim": 256, "freeze": True}

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
        stream_name=key, backbone_name=VISUAL_MODELS[key][1], pretrained=False,
        temporal_type=temporal_type, temporal_bidirectional=bidirectional,
        temporal_hidden=int(current["hidden"]), common_dim=int(current["dim"]),
        freeze_backbone=bool(current["freeze"]), grad_checkpointing=False,
        frame_chunk_size=0, num_frames=int(sticky.clip_settings()["n_frames"]),
    )


def render_config_controls(container, key: str, ns: str) -> dict:
    """The four architecture controls, writing back into the stored settings.

    `ns` namespaces the widget keys, so the hub and a subpage can both render the
    controls for the same model without colliding.
    """
    current = settings(key)
    labels = list(TEMPORAL)
    c1, c2 = container.columns(2)
    temporal = c1.selectbox("Temporal model", labels, key=f"{ns}_{key}_temporal",
                            index=labels.index(current["temporal"]))
    hidden = c1.slider("Temporal hidden", 64, 512, int(current["hidden"]), step=64,
                       key=f"{ns}_{key}_hidden", disabled=temporal == "Mean-pool",
                       help="Ignored when mean-pooling, which has no hidden state.")
    dim = c2.select_slider("Embedding dim", [128, 256, 512], int(current["dim"]),
                           key=f"{ns}_{key}_dim",
                           help="The width every stream is projected to before fusion.")
    freeze = c2.checkbox("Freeze backbone", value=bool(current["freeze"]),
                         key=f"{ns}_{key}_freeze")
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
    """Show the inherited clip read-only; return its path, or None."""
    clip = inherited_clip()
    if clip is None:
        container.info("No clip selected. Choose one on the **Preprocessing** page (a dataset "
                       "clip or your own upload) and it appears here.")
        return None
    clip_id, path = clip
    container.caption(f"Clip inherited from the Preprocessing page: **{clip_id}**")
    return path
