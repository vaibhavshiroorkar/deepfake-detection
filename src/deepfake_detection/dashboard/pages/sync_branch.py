from deepfake_detection.dashboard.components import render_page_header, render_status
from deepfake_detection.dashboard.navigation import PageState

render_page_header(
    "Planned evidence",
    "Sync branch",
    "See how mouth-audio alignment will form a temporal evidence stream.",
)
render_status(PageState.PROTOTYPE)
