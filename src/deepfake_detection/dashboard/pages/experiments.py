from deepfake_detection.dashboard.components import render_page_header, render_status
from deepfake_detection.dashboard.navigation import PageState

render_page_header(
    "Research record",
    "Experiments",
    "Review the saved development-validation record behind the visual baseline.",
)
render_status(PageState.READY)
