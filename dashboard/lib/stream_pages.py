"""Render functions for the three feature streams.

All three used to be separate sidebar pages; they now live as tabs on the single
Streams page (dashboard/pages/streams.py). Keeping the bodies here as functions
(rather than module-level page scripts) lets one page render all three tabs.

  * Visual  — Xception / EfficientNet as configurable model boxes (plus a
    not-yet-wired DINOv2 box). Real: builds and runs the config-driven stream.
  * Lip-Sync / Emotions — read-only scaffolds until Stages 4 / 5.

The dashboard never trains in-process (§7); Train tabs only emit the trainer
command, and Run forward-passes a clip with untrained weights.
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

from models.streams.common.config import StreamConfig, EFFICIENTNET_B0, XCEPTION, DINOV2
from dashboard.lib import stream_ui
from dashboard.lib.selectors import DATA_DIR, render_selection

TEMPORAL = {"BiLSTM": ("lstm", True), "GRU": ("gru", True), "Mean-pool": ("mean", False)}


def _visual_model_box(name: str, backbone_name: str, key: str, default_on: bool,
                      video_path) -> bool:
    with st.container(border=True):
        head, cfg = st.columns([1, 2])
        with head:
            enabled = st.toggle("Enable", value=default_on, key=f"{key}_enabled")
            st.markdown(f"### {name}")
            st.caption(f"`{backbone_name}`")
            st.caption("Included in fusion" if enabled else "Excluded from fusion")
        with cfg:
            c1, c2 = st.columns(2)
            temporal_label = c1.selectbox("Temporal model", list(TEMPORAL), key=f"{key}_temporal",
                                          disabled=not enabled)
            hidden = c1.slider("Temporal hidden", 64, 512, 256, step=64, key=f"{key}_hidden",
                               disabled=not enabled or temporal_label == "Mean-pool")
            common_dim = c2.select_slider("Embedding dim", [128, 256, 512], 256, key=f"{key}_dim",
                                          disabled=not enabled)
            freeze = c2.checkbox("Freeze backbone", value=True, key=f"{key}_freeze",
                                 disabled=not enabled)

        def make_config() -> StreamConfig:
            ttype, bidir = TEMPORAL[temporal_label]
            return StreamConfig(
                stream_name=key, backbone_name=backbone_name, pretrained=False,
                temporal_type=ttype, temporal_bidirectional=bidir, temporal_hidden=hidden,
                common_dim=common_dim, freeze_backbone=freeze, grad_checkpointing=False,
                frame_chunk_size=0,
            )

        train_tab, run_tab = st.tabs(["Train", "Run"])
        with train_tab:
            stream_ui.render_train_tab(st, key, key, backbone_name, enabled)
        with run_tab:
            stream_ui.render_run_tab(st, key, make_config, video_path, enabled)
    return enabled


def _visual_disabled_box(name: str, backbone_name: str, key: str, note: str):
    """A backbone that isn't wired into the stream yet — everything greyed out."""
    with st.container(border=True):
        head, cfg = st.columns([1, 2])
        with head:
            st.toggle("Enable", value=False, key=f"{key}_enabled", disabled=True)
            st.markdown(f"### {name}")
            st.caption(f"`{backbone_name}`")
            st.caption("Not wired yet")
        with cfg:
            c1, c2 = st.columns(2)
            c1.selectbox("Temporal model", list(TEMPORAL), key=f"{key}_temporal", disabled=True)
            c1.slider("Temporal hidden", 64, 512, 256, step=64, key=f"{key}_hidden", disabled=True)
            c2.select_slider("Embedding dim", [128, 256, 512], 256, key=f"{key}_dim", disabled=True)
            c2.checkbox("Freeze backbone", value=True, key=f"{key}_freeze", disabled=True)
        stream_ui.render_disabled_train_run(st, key, note)


def render_visual():
    st.caption("Artifact-focused backbones that read the face-crop sequence (never audio). "
               "Configure each, then Train (emits the background-trainer command — this page "
               "never trains, §7) or Run it on a clip to forward-pass a real sequence.")

    # One clip selection for the whole tab — every box's Run tab forward-passes it.
    with st.expander("Clip for Run inference (shared across boxes)", expanded=False):
        clip_row = render_selection()
    video_path = (str(DATA_DIR / clip_row["video_path"]) if clip_row is not None else None)

    st.session_state["stream_visual_enabled"] = {
        "efficientnet": _visual_model_box("EfficientNet-B0", EFFICIENTNET_B0, "m_effnet", True,
                                          video_path),
        "xception": _visual_model_box("Xception", XCEPTION, "m_xception", True, video_path),
    }
    _visual_disabled_box(
        "DINOv2 (ViT-S/14)", DINOV2, "m_dinov2",
        "Not wired into the visual stream yet — the DINOv2 backbone lands in a later stage.")


def _scaffold_stream(spec: dict, key_prefix: str, stage_note: str):
    st.info(spec["status"])
    st.caption(spec["note"])
    for model_name, desc in spec["models"]:
        with st.container(border=True):
            head, cfg = st.columns([1, 2])
            with head:
                st.toggle("Enable", value=True, key=f"{key_prefix}_{model_name}_enabled",
                          disabled=True)
                st.markdown(f"### {model_name}")
                st.caption(desc)
            with cfg:
                st.selectbox("Weights", ["(not downloaded)"],
                             key=f"{key_prefix}_{model_name}_weights", disabled=True)
                st.slider("Embedding dim", 128, 512, 256, step=64,
                          key=f"{key_prefix}_{model_name}_dim", disabled=True)
            stream_ui.render_disabled_train_run(st, f"{key_prefix}_{model_name}", stage_note)
    st.divider()
    st.caption(f"Read-only scaffold — the cross-attention model, its Train/Run controls and "
               f"metrics become live after {stage_note.rsplit('(', 1)[-1].rstrip(').')}. "
               f"This page never trains in-process (§7).")


def render_lipsync():
    from dashboard.lib.stream_spec import LIPSYNC_STREAM
    _scaffold_stream(LIPSYNC_STREAM, "lipsync",
                     "Available once the lip-sync stream is built (Stage 4).")


def render_emotions():
    from dashboard.lib.stream_spec import EMOTION_STREAM
    _scaffold_stream(EMOTION_STREAM, "emotion",
                     "Available once the emotion stream is built (Stage 5).")
