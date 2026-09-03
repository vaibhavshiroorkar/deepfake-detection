from deepfake_detection.dashboard.navigation import PAGES, PageState, page_by_slug


def test_dashboard_pages_follow_the_pipeline_order() -> None:
    assert [page.slug for page in PAGES] == [
        "overview",
        "video-input",
        "preprocessing",
        "visual-model",
        "prediction",
        "experiments",
        "audio-branch",
        "sync-branch",
        "fusion",
        "documentation",
    ]


def test_unfinished_research_pages_have_honest_states() -> None:
    assert page_by_slug("audio-branch").state is PageState.PROTOTYPE
    assert page_by_slug("sync-branch").state is PageState.PROTOTYPE
    assert page_by_slug("fusion").state is PageState.LOCKED
