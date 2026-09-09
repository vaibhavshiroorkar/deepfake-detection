"""Fetch the Localized Audio-Visual DeepFake dataset (LAV-DF) from Kaggle.

LAV-DF is the audiovisual corpus this project can actually obtain. DeepSpeak v2
is gated behind manual approval and AV-Deepfake1M behind an EULA, so neither can
be fetched on demand; LAV-DF is public on Kaggle and downloads immediately.

What it is: 136,304 videos, 36,431 real and 99,873 fake, built on VoxCeleb2. The
manipulations are content-driven rather than cosmetic. A word is chosen for
replacement by the largest swing it causes in perceived sentiment, text-to-speech
generates the new audio, and facial reenactment follows it, so the forgery
changes what the speaker appears to say.

Two properties make it unusually well suited to a cross-modal stream, and one
makes it awkward. Both are recorded here because they shape how it must be used.

  - **Localized labels.** Fake segments average 0.8 to 1.6 seconds inside an
    otherwise genuine clip, and the annotations say exactly where. That is a far
    stronger signal than a whole-clip label: it lets us ask whether the
    attention map lights up on the manipulated span, which validates the
    mechanism directly instead of by proxy.
  - **Both modalities are manipulated**, with audio present throughout, unlike
    Celeb-DF-v2 and MNW which carry no audio at all.
  - **Shared ancestry with FakeAVCeleb.** Both derive from VoxCeleb2, so
    identity overlap is possible and must be measured before LAV-DF is used as a
    cross-dataset test. As training data for the streams it is unaffected.

    uv run python scripts/fetch_lavdf.py --destination data/LAV-DF
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

DATASET = "elin75/localized-audio-visual-deepfake-dataset-lav-df"
# Verified by range request before this script was written.
EXPECTED_BYTES = 24_842_461_534
CHUNK = 8 * 1024 * 1024


def credentials() -> str:
    """Kaggle basic-auth header from the standard credentials file."""
    path = Path(os.path.expanduser("~/.kaggle/kaggle.json"))
    if not path.is_file():
        raise SystemExit(
            f"No Kaggle credentials at {path}. Create a token at "
            "https://www.kaggle.com/settings and save it there."
        )
    values = json.loads(path.read_text(encoding="utf-8"))
    pair = f"{values['username']}:{values['key']}".encode()
    return "Basic " + base64.b64encode(pair).decode()


def free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def download(archive: Path, auth: str) -> None:
    """Resume-capable download, because 24.8 GB over one connection will drop.

    Restarting from zero on a dropped connection is the difference between a
    retry costing minutes and costing hours, so the byte offset is sent as a
    Range header and the file is opened for append.
    """
    archive.parent.mkdir(parents=True, exist_ok=True)
    done = archive.stat().st_size if archive.exists() else 0
    if done >= EXPECTED_BYTES:
        print(f"archive already complete at {done:,} bytes")
        return

    headers = {"Authorization": auth}
    if done:
        headers["Range"] = f"bytes={done}-"
        print(f"resuming at {done:,} bytes ({100 * done / EXPECTED_BYTES:.1f}%)")

    request = urllib.request.Request(
        f"https://www.kaggle.com/api/v1/datasets/download/{DATASET}", headers=headers
    )
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
        mode = "ab" if done else "wb"
        with archive.open(mode) as handle:
            while True:
                block = response.read(CHUNK)
                if not block:
                    break
                handle.write(block)
                done += len(block)
                percent = 100 * done / EXPECTED_BYTES
                print(
                    f"\r  {done / 1e9:6.2f} / {EXPECTED_BYTES / 1e9:.2f} GB  "
                    f"{percent:5.1f}%",
                    end="",
                    flush=True,
                )
    print()


def extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.namelist()
        print(f"extracting {len(members):,} entries")
        for index, member in enumerate(members, start=1):
            bundle.extract(member, destination)
            if index % 2000 == 0:
                print(f"\r  {index:,} / {len(members):,}", end="", flush=True)
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("data/LAV-DF"))
    parser.add_argument("--archive", type=Path, default=None)
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the zip after extraction instead of deleting it.",
    )
    arguments = parser.parse_args(argv)

    archive = arguments.archive or arguments.destination.with_suffix(".zip")
    # The archive and its extraction coexist briefly, so both must fit.
    needed = EXPECTED_BYTES * 2
    available = free_bytes(Path("."))
    print(f"free disk {available / 1e9:.0f} GB, need about {needed / 1e9:.0f} GB")
    if available < needed:
        raise SystemExit(
            f"Not enough space: {available / 1e9:.0f} GB free, "
            f"{needed / 1e9:.0f} GB required."
        )

    download(archive, credentials())
    size = archive.stat().st_size
    if size != EXPECTED_BYTES:
        raise SystemExit(
            f"Archive is {size:,} bytes, expected {EXPECTED_BYTES:,}. "
            "Re-run to resume rather than extracting a truncated file."
        )

    extract(archive, arguments.destination)
    if not arguments.keep_archive:
        archive.unlink()
        print(f"removed {archive}")

    videos = sum(
        1 for path in arguments.destination.rglob("*") if path.suffix.lower() == ".mp4"
    )
    print(f"done: {videos:,} mp4 files under {arguments.destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
