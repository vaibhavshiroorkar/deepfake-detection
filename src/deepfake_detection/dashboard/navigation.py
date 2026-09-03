from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class PageState(StrEnum):
    READY = "ready"
    PROTOTYPE = "prototype"
    LOCKED = "locked"


@dataclass(frozen=True, slots=True)
class PageSpec:
    slug: str
    module: str
    title: str
    state: PageState

    @property
    def navigation_label(self) -> str:
        if self.state is PageState.READY:
            return self.title
        return f"{self.title} ({self.state.value})"


PAGES = (
    PageSpec("overview", "pages/overview.py", "Overview", PageState.READY),
    PageSpec("video-input", "pages/video_input.py", "1. Video input", PageState.READY),
    PageSpec(
        "preprocessing", "pages/preprocessing.py", "2. Preprocessing", PageState.READY
    ),
    PageSpec(
        "visual-model", "pages/visual_model.py", "3. Visual model", PageState.READY
    ),
    PageSpec("prediction", "pages/prediction.py", "4. Prediction", PageState.READY),
    PageSpec("experiments", "pages/experiments.py", "Experiments", PageState.READY),
    PageSpec(
        "audio-branch", "pages/audio_branch.py", "Audio branch", PageState.PROTOTYPE
    ),
    PageSpec("sync-branch", "pages/sync_branch.py", "Sync branch", PageState.PROTOTYPE),
    PageSpec("fusion", "pages/fusion.py", "Fusion", PageState.LOCKED),
    PageSpec(
        "documentation", "pages/documentation.py", "Documentation", PageState.READY
    ),
)


def page_by_slug(slug: str) -> PageSpec:
    for page in PAGES:
        if page.slug == slug:
            return page
    raise KeyError(slug)
