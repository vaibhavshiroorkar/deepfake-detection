from pathlib import Path
from urllib.parse import urlparse

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


def test_documentation_page_links_to_tracked_records_on_github() -> None:
    page = AppTest.from_file(
        "src/deepfake_detection/dashboard/pages/documentation.py"
    ).run()

    assert not page.exception
    expected_base = "https://github.com/vaibhavshiroorkar/deepfake-detection/blob/main/"
    actual_targets = [
        item.value.split("(", 1)[1][:-1]
        for item in page.markdown
        if item.value.startswith("[")
    ]
    assert actual_targets == [target for _, target in DOCUMENTS]
    for target in actual_targets:
        parsed = urlparse(target)
        assert target.startswith(expected_base)
        assert parsed.scheme == "https"
        assert parsed.netloc == "github.com"


def test_experiments_page_reports_missing_local_evidence_without_a_crash() -> None:
    page = AppTest.from_file(
        "src/deepfake_detection/dashboard/pages/experiments.py"
    ).run()

    assert not page.exception
    assert "Evidence is unavailable" in _page_body(page)


def test_experiments_page_reports_an_evidence_read_os_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deepfake_detection.dashboard import evidence

    def fail(metrics_path: Path, history_path: Path) -> object:
        raise OSError("evidence device is unavailable")

    monkeypatch.setattr(evidence, "load_validation_evidence", fail)

    page = AppTest.from_file(
        "src/deepfake_detection/dashboard/pages/experiments.py"
    ).run()

    assert not page.exception
    assert "Evidence is unavailable" in _page_body(page)
