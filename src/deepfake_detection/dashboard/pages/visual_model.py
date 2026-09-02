from deepfake_detection.dashboard.components import render_page_header, render_status
from deepfake_detection.dashboard.navigation import PageState

render_page_header(
    "Stage 3",
    "3. Visual model",
    "Trace frame features through the frozen visual classifier.",
)
render_status(PageState.READY)
