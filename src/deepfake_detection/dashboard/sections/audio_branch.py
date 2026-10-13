import streamlit as st

from deepfake_detection.dashboard.components import render_page_header, render_status
from deepfake_detection.dashboard.status import PageState


def render_audio_branch(*, embedded: bool = False) -> None:
    if not embedded:
        render_page_header(
            "Planned evidence",
            "Audio branch",
            "See what remains before audio spoof evidence can support a research claim.",
        )
    render_status(PageState.PROTOTYPE)

    st.subheader("Prototype teaching path")
    st.write("Input: a four-second, 16 kHz waveform with shape [B, S].")
    st.write(
        "Processing: Wav2Vec2 Base produces temporal tokens. A projection, learned "
        "attention pool, and linear head produce an audio-spoof logit."
    )
    st.write("Output: one uncalibrated audio logit and a clip embedding.")
    st.info(
        "Trained at three seeds, and at chance cross-corpus: ROC-AUC 0.5054 "
        "[0.4747, 0.5371] on DFDC against 0.7739 in-domain (F30)."
    )
    st.write(
        "No checkpoint is loaded here and this page does not calculate a probability."
    )
    st.write(
        "Unlock condition: a corpus with manipulated audio that is not built on "
        "VoxCeleb2, so the branch can be scored on capture it has not seen."
    )


if __name__ == "__main__":
    render_audio_branch()
