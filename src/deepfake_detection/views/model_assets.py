from __future__ import annotations

import hashlib
import os
from pathlib import Path
from urllib.request import Request, urlopen
from uuid import uuid4

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/"
    "47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_detection_yunet/"
    "face_detection_yunet_2026may.onnx"
)
YUNET_SIZE_BYTES = 229738
YUNET_SHA256 = "ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0"


def fetch_yunet_model(destination: Path, *, force: bool = False) -> Path:
    resolved = destination.expanduser().resolve()
    if resolved.exists():
        if _has_expected_integrity(resolved):
            return resolved
        if not force:
            raise ValueError(
                "Existing YuNet model failed integrity checks. "
                "Pass force=True to replace it."
            )

    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.{uuid4().hex}.tmp")
    request = Request(YUNET_URL, headers={"User-Agent": "deepfake-generalization"})
    try:
        with urlopen(request, timeout=60) as response:  # noqa: S310
            with temporary.open("xb") as output:
                byte_limit = YUNET_SIZE_BYTES + 1
                downloaded = 0
                while chunk := response.read(min(1024 * 1024, byte_limit - downloaded)):
                    downloaded += len(chunk)
                    if downloaded > YUNET_SIZE_BYTES:
                        raise ValueError(
                            "Downloaded YuNet model failed integrity checks: "
                            "size limit exceeded"
                        )
                    output.write(chunk)
                    if downloaded == byte_limit:
                        break
                output.flush()
                os.fsync(output.fileno())
        if not _has_expected_integrity(temporary):
            raise ValueError("Downloaded YuNet model failed integrity checks")
        temporary.replace(resolved)
    finally:
        temporary.unlink(missing_ok=True)
    return resolved


def _has_expected_integrity(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size != YUNET_SIZE_BYTES:
        return False
    with path.open("rb") as model_file:
        digest = hashlib.file_digest(model_file, "sha256").hexdigest()
    return digest == YUNET_SHA256
