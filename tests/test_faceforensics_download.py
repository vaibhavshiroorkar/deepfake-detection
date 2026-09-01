from pathlib import Path

import pytest

from deepfake_detection.data import download


def test_download_file_removes_partial_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*_: object, **__: object) -> None:
        raise OSError("transfer failed")

    monkeypatch.setattr(download.urllib.request, "urlretrieve", fail)

    with pytest.raises(OSError, match="transfer failed"):
        download.download_file("https://example.test/video.mp4", tmp_path / "video.mp4")

    assert tuple(tmp_path.iterdir()) == ()


def test_download_file_preserves_existing_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "video.mp4"
    target.write_bytes(b"complete")

    def unexpected(*_: object, **__: object) -> None:
        raise AssertionError("existing payload must not be downloaded")

    monkeypatch.setattr(download.urllib.request, "urlretrieve", unexpected)

    download.download_file("https://example.test/video.mp4", target)

    assert target.read_bytes() == b"complete"
