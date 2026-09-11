"""Fetch FaceForensics++ c23, the corpus the cross-dataset protocol trains on.

Why this dataset and not another: nearly every cross-dataset table in the
literature trains on FF++ c23 and reports zero-shot AUC on Celeb-DF-v2 and
DFDC. This project trains on FakeAVCeleb, which almost nobody uses as a training
corpus, so none of its numbers are comparable to a published one. Adding FF++ is
what makes the comparison possible at all.

PROVENANCE WARNING. FaceForensics++ is officially distributed under a EULA with
TUM: you complete their form and they send a download script. This pulls from a
third-party Hugging Face mirror that states no licence. The data is
research-available either way, but a data card should record how it was actually
obtained, and "third-party mirror" is weaker than "signed EULA". Prefer the
official route when there is time for it.

Resumable: the mirror serves range requests, so a 17.9 GB transfer that drops
picks up where it stopped rather than starting again.

    uv run python scripts/fetch_ffpp.py --destination data/FaceForensics++
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path

REPO = "bitmind/FaceForensicsC23"
ARCHIVE = "FaceForensics%2B%2B_C23.zip"
URL = f"https://huggingface.co/datasets/{REPO}/resolve/main/{ARCHIVE}"
CHUNK = 1 << 20


def _token() -> str | None:
    """The Hugging Face token, from the environment or a local .env."""
    found = os.environ.get("HF_TOKEN")
    if found:
        return found
    env = Path(".env")
    if not env.is_file():
        return None
    for line in env.read_text(encoding="utf-8").splitlines():
        if line.startswith("HF_TOKEN"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _request(headers: dict, method: str = "GET") -> urllib.request.Request:
    return urllib.request.Request(URL, headers=headers, method=method)


def download(target: Path, token: str | None) -> Path:
    """Stream the archive to `target`, resuming an interrupted transfer."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    with urllib.request.urlopen(  # noqa: S310
        _request(headers, "HEAD"), timeout=60
    ) as head:
        total = int(head.headers.get("Content-Length", 0))
    target.parent.mkdir(parents=True, exist_ok=True)

    done = target.stat().st_size if target.is_file() else 0
    if total and done >= total:
        print(f"already complete: {target} ({done / 1e9:.2f} GB)")
        return target
    if done:
        # A partial file is resumed rather than restarted. The mirror advertises
        # Accept-Ranges: bytes, and 17.9 GB is far too much to redo.
        print(f"resuming at {done / 1e9:.2f} GB of {total / 1e9:.2f} GB")
        headers["Range"] = f"bytes={done}-"

    started = time.perf_counter()
    with urllib.request.urlopen(  # noqa: S310
        _request(headers), timeout=120
    ) as response:
        mode = "ab" if done else "wb"
        with target.open(mode) as handle:
            last = started
            while chunk := response.read(CHUNK):
                handle.write(chunk)
                done += len(chunk)
                now = time.perf_counter()
                if now - last >= 10:
                    rate = done / max(now - started, 1e-9) / 1e6
                    percent = f"{done / total:.1%}" if total else "?"
                    print(
                        f"  {done / 1e9:6.2f} GB  {percent:>6}  {rate:5.1f} MB/s",
                        flush=True,
                    )
                    last = now
    print(f"downloaded {done / 1e9:.2f} GB to {target}")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("data/FaceForensics++"))
    parser.add_argument(
        "--archive-name", default="FaceForensics++_C23.zip", help="Local file name."
    )
    arguments = parser.parse_args(argv)

    token = _token()
    if token is None:
        print("No HF_TOKEN found; the mirror may still allow anonymous access.")

    target = arguments.destination / arguments.archive_name
    try:
        download(target, token)
    except (OSError, ValueError) as error:
        # A partial file is left in place on purpose: the next run resumes it.
        print(f"download failed: {type(error).__name__}: {error}")
        print("rerun this command to resume from where it stopped")
        return 1
    print(
        "\nProvenance: third-party mirror, no licence stated. Record that in "
        "docs/data-card.md rather than implying a signed EULA."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
