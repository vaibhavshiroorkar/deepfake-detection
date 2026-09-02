from deepfake_detection.dashboard.components import render_page_header, render_status
from deepfake_detection.dashboard.navigation import PageState

render_page_header(
    "Stage 1",
    "1. Video input",
    "Select the local talking-head clip used by every later pipeline stage.",
)
render_status(PageState.READY)
