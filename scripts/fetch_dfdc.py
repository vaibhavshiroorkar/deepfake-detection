"""Fetch a slice of the DFDC training set from a Kaggle mirror.

DFDC is the cross-dataset test this project actually needs. Everything else on
hand shares ancestry or lacks a modality:

  - FakeAVCeleb and LAV-DF are both built on VoxCeleb2, so evaluating one on the
    other is not a corpus change, only a generator change.
  - Celeb-DF-v2 and MNW carry no audio track at all, so neither can score an
    audiovisual stream.

DFDC was filmed for the challenge with 3,426 paid actors, has audio, and
includes audio-swapped forgeries. It is the only corpus available here that is
independent of VoxCeleb2 and complete in both modalities.

The mirror is a single 103 GB zip, and only a slice of it is wanted. Rather than
pull all of it, this reads the zip64 central directory over HTTP range requests,
locates the requested parts, and downloads just those bytes. Parts are stored
contiguously from the start of the archive, so one part is a prefix and two
parts is a slightly longer prefix.

    uv run python scripts/fetch_dfdc.py --parts 2 --destination data/DFDC
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import struct
import sys
import urllib.request
import zlib
from pathlib import Path

DATASET = "pranay22077/dfdc-10"
TOTAL_BYTES = 103_589_478_799
CHUNK = 8 * 1024 * 1024


def credentials() -> str:
    path = Path(os.path.expanduser("~/.kaggle/kaggle.json"))
    if not path.is_file():
        raise SystemExit(f"No Kaggle credentials at {path}")
    values = json.loads(path.read_text(encoding="utf-8"))
    return "Basic " + base64.b64encode(
        f"{values['username']}:{values['key']}".encode()
    ).decode()


def ranged(auth: str, start: int, end: int) -> bytes:
    request = urllib.request.Request(
        f"https://www.kaggle.com/api/v1/datasets/download/{DATASET}",
        headers={"Authorization": auth, "Range": f"bytes={start}-{end}"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:  # noqa: S310
        return response.read()


def central_directory(auth: str) -> list[tuple[str, int, int, int]]:
    """Every entry as (name, local header offset, compressed, uncompressed).

    The archive is zip64, so the 32-bit size and offset fields are saturated to
    0xFFFFFFFF and the real values live in the extra field. Reading only the
    32-bit fields yields nonsense offsets, which is worth stating because the
    failure is silent: you get plausible-looking numbers that address the wrong
    bytes.
    """
    tail = ranged(auth, TOTAL_BYTES - 65536, TOTAL_BYTES - 1)
    marker = tail.rfind(b"PK\x06\x06")
    if marker < 0:
        raise SystemExit("No zip64 end-of-central-directory record found")
    size, offset = struct.unpack_from("<QQ", tail, marker + 40)
    blob = ranged(auth, offset, offset + size - 1)

    entries: list[tuple[str, int, int, int]] = []
    cursor = 0
    while cursor < len(blob) and blob[cursor : cursor + 4] == b"PK\x01\x02":
        compressed, uncompressed = struct.unpack_from("<II", blob, cursor + 20)
        name_len, extra_len, comment_len = struct.unpack_from("<HHH", blob, cursor + 28)
        header_offset = struct.unpack_from("<I", blob, cursor + 42)[0]
        name = blob[cursor + 46 : cursor + 46 + name_len].decode("utf-8", "replace")
        extra = blob[cursor + 46 + name_len : cursor + 46 + name_len + extra_len]
        position = 0
        while position + 4 <= len(extra):
            field_id, field_size = struct.unpack_from("<HH", extra, position)
            if field_id == 0x0001:
                at = position + 4
                if uncompressed == 0xFFFFFFFF:
                    uncompressed = struct.unpack_from("<Q", extra, at)[0]
                    at += 8
                if compressed == 0xFFFFFFFF:
                    compressed = struct.unpack_from("<Q", extra, at)[0]
                    at += 8
                if header_offset == 0xFFFFFFFF:
                    header_offset = struct.unpack_from("<Q", extra, at)[0]
                break
            position += 4 + field_size
        entries.append((name, header_offset, compressed, uncompressed))
        cursor += 46 + name_len + extra_len + comment_len
    return entries


def download_prefix(auth: str, archive: Path, end: int) -> None:
    """Resume-capable prefix download; a multi-gigabyte transfer will drop."""
    archive.parent.mkdir(parents=True, exist_ok=True)
    done = archive.stat().st_size if archive.exists() else 0
    if done >= end:
        print(f"prefix already present: {done:,} bytes")
        return
    print(f"downloading bytes {done:,} to {end:,} ({(end - done) / 1e9:.2f} GB)")
    with archive.open("ab" if done else "wb") as handle:
        while done < end:
            stop = min(done + 256 * 1024 * 1024, end) - 1
            handle.write(ranged(auth, done, stop))
            done = archive.stat().st_size
            print(f"\r  {done / 1e9:6.2f} / {end / 1e9:.2f} GB", end="", flush=True)
    print()


def extract(archive: Path, entries, destination: Path) -> int:
    """Pull each wanted entry out of the prefix by its own byte offset.

    `zipfile` cannot open a truncated archive because it looks for the central
    directory at the end, which was left behind. Each local header carries the
    name and its own sizes, so entries are read individually instead.
    """
    destination.mkdir(parents=True, exist_ok=True)
    written = 0
    with archive.open("rb") as handle:
        for name, offset, compressed, _ in entries:
            handle.seek(offset)
            header = handle.read(30)
            if header[:4] != b"PK\x03\x04":
                continue
            method = struct.unpack_from("<H", header, 8)[0]
            name_len, extra_len = struct.unpack_from("<HH", header, 26)
            handle.seek(offset + 30 + name_len + extra_len)
            payload = handle.read(compressed)
            if len(payload) < compressed:
                break
            if method == 8:
                payload = zlib.decompressobj(-zlib.MAX_WBITS).decompress(payload)
            # Videos flatten safely because DFDC names them with random
            # 10-character strings, but every part ships a `metadata.json` and
            # flattening those loses all but the last. The part is kept in the
            # name so each part's labels survive.
            leaf = Path(name).name
            if leaf.startswith("metadata"):
                leaf = f"metadata-{name.split('/')[0]}.json"
            target = destination / leaf
            target.write_bytes(payload)
            written += 1
            if written % 200 == 0:
                print(f"\r  extracted {written:,}", end="", flush=True)
    print(f"\r  extracted {written:,}")
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", type=int, default=2)
    parser.add_argument("--destination", type=Path, default=Path("data/DFDC"))
    parser.add_argument("--archive", type=Path, default=None)
    parser.add_argument("--keep-archive", action="store_true")
    arguments = parser.parse_args(argv)

    auth = credentials()
    print("reading the archive index over range requests")
    entries = central_directory(auth)
    names = sorted({name.split("/")[0] for name, *_ in entries})
    wanted_parts = names[: arguments.parts]
    wanted = [e for e in entries if e[0].split("/")[0] in wanted_parts]
    end = max(offset + compressed for _, offset, compressed, _ in wanted)
    print(f"parts: {', '.join(wanted_parts)}")
    print(f"{len(wanted):,} files, prefix of {end / 1e9:.2f} GB of {TOTAL_BYTES / 1e9:.0f} GB")

    free = shutil.disk_usage(".").free
    if free < end * 2:
        raise SystemExit(
            f"Not enough space: {free / 1e9:.0f} GB free, {end * 2 / 1e9:.0f} GB needed"
        )

    archive = arguments.archive or arguments.destination.with_suffix(".partial.zip")
    download_prefix(auth, archive, end)
    count = extract(archive, wanted, arguments.destination)
    if not arguments.keep_archive:
        archive.unlink()
        print(f"removed {archive}")
    videos = sum(1 for p in arguments.destination.iterdir() if p.suffix == ".mp4")
    print(f"done: {count:,} files, {videos:,} mp4 under {arguments.destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
