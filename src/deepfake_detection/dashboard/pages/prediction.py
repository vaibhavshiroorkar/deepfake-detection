from deepfake_detection.dashboard.components import render_page_header, render_status
from deepfake_detection.dashboard.navigation import PageState

render_page_header(
    "Stage 4",
    "4. Prediction",
    "Run the frozen visual baseline and read its evidence limits.",
)
render_status(PageState.READY)
