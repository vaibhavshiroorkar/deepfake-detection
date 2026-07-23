"""Visual streams — Xception and EfficientNet as configurable model boxes.

Each box toggles the backbone in/out of the (future) fusion set and configures
it (temporal model, hidden size, freezing, embedding dim, pretrained weights).
"Build & inspect" instantiates the real config-driven VisualStream and forward-
passes a dummy clip to report its shapes and parameter counts — inference only,
no training (training is a background script, PROJECT_OVERVIEW.md §7).
"""
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

from models.streams.common.config import StreamConfig, EFFICIENTNET_B0, XCEPTION

st.title("Visual streams")
st.caption("Artifact-focused backbones that read the face-crop sequence (never audio). "
           "Toggle and configure each, then build it to confirm the wiring. Training runs "
           "as a background script — this page never trains (§7).")

TEMPORAL = {"BiLSTM": ("lstm", True), "GRU": ("gru", True), "Mean-pool": ("mean", False)}


def model_box(name: str, backbone_name: str, key: str, default_on: bool):
    with st.container(border=True):
        head, cfg = st.columns([1, 2])
        with head:
            enabled = st.toggle("Enable", value=default_on, key=f"{key}_enabled")
            st.markdown(f"### {name}")
            st.caption(f"`{backbone_name}`")
            st.caption("Included in fusion" if enabled else "Excluded from fusion")
        with cfg:
            c1, c2 = st.columns(2)
            temporal_label = c1.selectbox("Temporal model", list(TEMPORAL), key=f"{key}_temporal")
            hidden = c1.slider("Temporal hidden", 64, 512, 256, step=64, key=f"{key}_hidden",
                               disabled=temporal_label == "Mean-pool")
            common_dim = c2.select_slider("Embedding dim", [128, 256, 512], 256, key=f"{key}_dim")
            freeze = c2.checkbox("Freeze backbone", value=True, key=f"{key}_freeze")
            pretrained = c2.checkbox("Pretrained weights (downloads on first build)",
                                     value=False, key=f"{key}_pretrained")
            build = st.button("Build & inspect", key=f"{key}_build", disabled=not enabled)

        if build:
            ttype, bidir = TEMPORAL[temporal_label]
            config = StreamConfig(
                stream_name=key, backbone_name=backbone_name, pretrained=pretrained,
                temporal_type=ttype, temporal_bidirectional=bidir, temporal_hidden=hidden,
                common_dim=common_dim, freeze_backbone=freeze, grad_checkpointing=False,
                frame_chunk_size=0,
            )
            with st.spinner(f"Building {name}…"):
                import torch
                from models.streams.common.visual_stream import build_visual_stream
                model = build_visual_stream(config).eval()
                with torch.no_grad():
                    logit, emb = model(torch.randn(1, config.num_frames, 3, 224, 224))
                counts = model.param_counts()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Backbone features", counts["feature_dim"])
            m2.metric("Embedding dim", counts["embedding_dim"])
            m3.metric("Total params", f"{counts['total'] / 1e6:.1f}M")
            m4.metric("Trainable params", f"{counts['trainable'] / 1e6:.1f}M")
            st.code(
                f"input  : [1, {config.num_frames}, 3, 224, 224]\n"
                f"logit  : {tuple(logit.shape)}  (dev head, discarded at fusion)\n"
                f"embed  : {tuple(emb.shape)}  -> feature store -> fusion",
                language="text",
            )
        return enabled


st.session_state["stream_visual_enabled"] = {
    "efficientnet": model_box("EfficientNet-B0", EFFICIENTNET_B0, "m_effnet", True),
    "xception": model_box("Xception", XCEPTION, "m_xception", True),
}
