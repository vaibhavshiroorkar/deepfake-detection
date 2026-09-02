from deepfake_detection.dashboard.components import (
    render_page_header,
    render_status,
    require_upload,
)
from deepfake_detection.dashboard.navigation import PageState

render_page_header(
    "Stage 2",
    "2. Preprocessing",
    "See how one clip becomes a tracked and normalized face sequence.",
)
render_status(PageState.READY)
require_upload()
