from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from deepfake_detection.dashboard.pages.documentation import DOCUMENTS


def _page_body(page: AppTest) -> str:
    elements = (*page.markdown, *page.info, *page.error, *page.caption)
    return " ".join(item.value for item in elements)


@pytest.mark.parametrize(
    ("page_path", "required_text"),
    (
        ("audio_branch.py", ("prototype", "full training is incomplete")),
        ("sync_branch.py", ("prototype", "full training is incomplete")),
        ("fusion.py", ("locked", "software fixture")),
    ),
)
def test_status_pages_name_their_evidence_limits(
    page_path: str, required_text: tuple[str, ...]
) -> None:
    page = AppTest.from_file(
        f"src/deepfake_detection/dashboard/pages/{page_path}"
    ).run()

    assert not page.exception
    body = _page_body(page).lower()
    for text in required_text:
        assert text in body


def test_documentation_page_links_only_to_existing_files() -> None:
    page = AppTest.from_file(
        "src/deepfake_detection/dashboard/pages/documentation.py"
    ).run()

    assert not page.exception
    assert all(Path(relative).is_file() for _, relative in DOCUMENTS)
    body = _page_body(page)
    for _, relative in DOCUMENTS:
        assert relative in body


def test_experiments_page_reports_missing_local_evidence_without_a_crash() -> None:
    page = AppTest.from_file(
        "src/deepfake_detection/dashboard/pages/experiments.py"
    ).run()

    assert not page.exception
    assert "Evidence is unavailable" in _page_body(page)
