from deepfake_detection.dashboard.components import render_page_header, render_status
from deepfake_detection.dashboard.navigation import PageState

render_page_header(
    "Evidence gate",
    "Fusion",
    "See how calibrated visual, audio, and sync evidence will enter late fusion.",
)
render_status(PageState.LOCKED)
