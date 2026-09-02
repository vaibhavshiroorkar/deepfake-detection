from deepfake_detection.dashboard.components import render_page_header, render_status
from deepfake_detection.dashboard.navigation import PageState

render_page_header(
    "Project map",
    "Overview",
    "Follow the detector from an uploaded clip to an evidence-limited prediction.",
)
render_status(PageState.READY)
