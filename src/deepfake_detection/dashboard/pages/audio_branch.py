from deepfake_detection.dashboard.components import render_page_header, render_status
from deepfake_detection.dashboard.navigation import PageState

render_page_header(
    "Planned evidence",
    "Audio branch",
    "See what remains before audio spoof evidence can support a research claim.",
)
render_status(PageState.PROTOTYPE)
