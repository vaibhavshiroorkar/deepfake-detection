"""The file check: labels that AI tools write into a file on purpose.

Two kinds are read. An AIGC metadata tag whose Label is "1", which Seedance 1.0 (Dreamina) writes into
every file it exports: all 394 of its HiFi-AVDF clips carry one, and none of the 1,905 real clips does
(checked 11-18). And a C2PA manifest whose digital source type is the IPTC term for media made by a
trained model, "trainedAlgorithmicMedia".

Nothing else about the file is evidence. Codec, bitrate, length and encoder strings describe how a
file was saved, and the shortcut audit shows that in research data they follow the label for reasons
that have nothing to do with the fake. A missing label proves nothing: an upload or a re-save strips
it, as it did for every other HiFi-AVDF generator.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

AI_SOURCE_TYPE = b"digitalsourcetype/trainedAlgorithmicMedia"
MAX_BYTES = 128 << 20  # uploads are capped well below this; read no more than it


# Makers a C2PA manifest may name. Only long names: a short one ("Veo", "Sora") could turn up by chance
# in compressed video bytes and name the wrong tool.
C2PA_MAKERS = (
    b"OpenAI",
    b"Google",
    b"Adobe",
    b"Microsoft",
    b"Runway",
    b"Midjourney",
    b"ByteDance",
)
TOOL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._-]{0,39}")


def _json(value) -> dict:
    try:
        found = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return found if isinstance(found, dict) else {}


def _named_tool(tags: dict[str, str]) -> str | None:
    """The tool a file names in its UserComment (Dreamina writes "source_info"), if it is plain text."""
    for name, value in tags.items():
        if name.lower() == "usercomment":
            tool = str(_json(value).get("source_info", "")).strip()
            if TOOL_NAME.fullmatch(tool):
                return tool
    return None


def labels_from_tags(tags: dict[str, str]) -> list[str]:
    """Plain-language descriptions of every AI label in the file's metadata tags."""
    found = []
    tool = _named_tool(tags)
    for name, value in tags.items():
        if name.lower() != "aigc":
            continue
        if str(_json(value).get("Label", "")).strip() == "1":
            text = "an AIGC label marking the file as AI-generated content"
            found.append(text + (f', which names the tool "{tool}"' if tool else ""))
    return found


def labels_from_bytes(data: bytes) -> list[str]:
    """A C2PA manifest that declares the media was made by a trained model, with any known maker it names."""
    if b"c2pa" not in data or AI_SOURCE_TYPE not in data:
        return []
    makers = [m.decode() for m in C2PA_MAKERS if m in data]
    text = "a C2PA content credential declaring AI-generated media"
    if makers:
        text += ", which names " + " and ".join(f'"{m}"' for m in makers)
    return [text]


def _tags(path: Path) -> dict[str, str]:
    run = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format_tags:stream_tags",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    data = json.loads(run.stdout or "{}")
    tags = dict(data.get("format", {}).get("tags", {}))
    for stream in data.get("streams", []):
        tags.update(stream.get("tags", {}))
    return tags


def read(path: Path) -> list[str]:
    """Every AI label in the file at `path`, in plain words; empty when there is none."""
    with open(path, "rb") as f:
        data = f.read(MAX_BYTES)
    return labels_from_tags(_tags(path)) + labels_from_bytes(data)
