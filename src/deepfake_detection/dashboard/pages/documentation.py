from deepfake_detection.dashboard.components import render_page_header, render_status
from deepfake_detection.dashboard.navigation import PageState

render_page_header(
    "Project record",
    "Documentation",
    "Open the repository records that define scope, data, runs, and reproducibility.",
)
render_status(PageState.READY)
