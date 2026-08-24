from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from deepfake_detection.views import model_assets


class ByteResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.offset = 0

    def __enter__(self) -> ByteResponse:
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self.payload) - self.offset
        chunk = self.payload[self.offset : self.offset + size]
        self.offset += len(chunk)
        return chunk


def configure_test_asset(monkeypatch: pytest.MonkeyPatch, payload: bytes) -> None:
    monkeypatch.setattr(model_assets, "YUNET_SIZE_BYTES", len(payload))
    monkeypatch.setattr(
        model_assets, "YUNET_SHA256", hashlib.sha256(payload).hexdigest()
    )


def test_yunet_asset_manifest_pins_the_fetch_contract() -> None:
    manifest_path = (
        Path(__file__).parents[1] / "configs" / "assets" / "yunet-2026may.json"
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest == {
        "asset_id": "opencv-zoo-yunet-2026may",
        "source_commit": "47534e27c9851bb1128ccc0102f1145e27f23f98",
        "url": model_assets.YUNET_URL,
        "size_bytes": 229738,
        "sha256": "ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0",
    }


def test_fetch_yunet_downloads_fixed_url_and_reuses_valid_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"pinned-yunet-model"
    configure_test_asset(monkeypatch, payload)
    calls: list[tuple[str, int]] = []

    def fake_urlopen(request: Any, timeout: int) -> ByteResponse:
        calls.append((request.full_url, timeout))
        return ByteResponse(payload)

    monkeypatch.setattr(model_assets, "urlopen", fake_urlopen)
    destination = tmp_path / "models" / "yunet.onnx"

    first = model_assets.fetch_yunet_model(destination)
    second = model_assets.fetch_yunet_model(destination)

    assert first == destination.resolve()
    assert second == destination.resolve()
    assert destination.read_bytes() == payload
    assert calls == [(model_assets.YUNET_URL, 60)]
    assert list(destination.parent.glob("*.tmp")) == []


def test_fetch_yunet_rejects_wrong_existing_file_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"pinned-yunet-model"
    configure_test_asset(monkeypatch, payload)
    destination = tmp_path / "yunet.onnx"
    destination.write_bytes(b"wrong")

    def unexpected_urlopen(request: Any, timeout: int) -> ByteResponse:
        raise AssertionError("A wrong existing asset must not trigger a download")

    monkeypatch.setattr(model_assets, "urlopen", unexpected_urlopen)

    with pytest.raises(ValueError, match="force=True"):
        model_assets.fetch_yunet_model(destination)

    assert destination.read_bytes() == b"wrong"


def test_fetch_yunet_force_replaces_wrong_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"pinned-yunet-model"
    configure_test_asset(monkeypatch, payload)
    destination = tmp_path / "yunet.onnx"
    destination.write_bytes(b"wrong")
    monkeypatch.setattr(
        model_assets,
        "urlopen",
        lambda request, timeout: ByteResponse(payload),
    )

    model_assets.fetch_yunet_model(destination, force=True)

    assert destination.read_bytes() == payload


@pytest.mark.parametrize("download", [b"wrong-size", b"XXXXXXXXXXXXXXXXXXX"])
def test_fetch_yunet_rejects_bad_download_without_replacing_destination(
    download: bytes, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = b"expected-hash-value"
    configure_test_asset(monkeypatch, expected)
    destination = tmp_path / "yunet.onnx"
    destination.write_bytes(b"existing-wrong-file")
    monkeypatch.setattr(
        model_assets,
        "urlopen",
        lambda request, timeout: ByteResponse(download),
    )

    with pytest.raises(ValueError, match="integrity"):
        model_assets.fetch_yunet_model(destination, force=True)

    assert destination.read_bytes() == b"existing-wrong-file"
    assert list(tmp_path.glob("*.tmp")) == []
