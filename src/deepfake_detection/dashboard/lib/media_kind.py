"""What kind of media a file is, and which streams can read it.

A video carries a face track and a soundtrack, so every stream can run on it. An
image has no time axis and no audio. A sound file has no face. Routing by file
kind is what stops the dashboard from asking a lip-sync model to read a JPEG and
then reporting whatever number falls out.

The rule this encodes: a stream runs only when every view it needs is present.
That is the same abstention policy the batch pipeline uses, where a clip with no
stable face track gets no visual view and is reported rather than dropped.

An image is the awkward case and is treated honestly. The visual stream takes a
sequence and an image is a sequence of one, which runs but is not what the model
was trained on: it learned over sixteen frames spanning a whole clip, and the
frame-to-frame variation it reads is exactly what a single frame cannot show.
`ROUTING` marks that as degraded rather than unavailable, and the caller is
expected to say so.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"})
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"})
AUDIO_SUFFIXES = frozenset({".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"})

VIDEO = "video"
IMAGE = "image"
AUDIO = "audio"
UNKNOWN = "unknown"

UPLOAD_SUFFIXES = tuple(
    sorted(suffix.lstrip(".") for suffix in VIDEO_SUFFIXES | IMAGE_SUFFIXES | AUDIO_SUFFIXES)
)


@dataclass(frozen=True, slots=True)
class StreamAvailability:
    """Whether one stream can read this media, and why not when it cannot."""

    stream: str
    available: bool
    degraded: bool = False
    reason: str = ""


# Which streams each media kind can drive. The reasons are written for a reader
# who does not know the architecture: they say what is missing, not which tensor
# is None.
ROUTING: dict[str, dict[str, StreamAvailability]] = {
    VIDEO: {
        "visual": StreamAvailability("visual", True),
        "audio": StreamAvailability("audio", True),
        "lipsync": StreamAvailability("lipsync", True),
        "emotion": StreamAvailability("emotion", True),
    },
    IMAGE: {
        "visual": StreamAvailability(
            "visual",
            True,
            degraded=True,
            reason=(
                "Runs on one frame. The model was trained on sixteen frames from "
                "across a clip, and part of what it reads is how the face changes "
                "between them, which a single image cannot show."
            ),
        ),
        "audio": StreamAvailability("audio", False, reason="No sound track."),
        "lipsync": StreamAvailability(
            "lipsync", False, reason="Needs mouth movement and sound together."
        ),
        "emotion": StreamAvailability(
            "emotion", False, reason="Needs a face and a voice together."
        ),
    },
    AUDIO: {
        "visual": StreamAvailability("visual", False, reason="No picture."),
        "audio": StreamAvailability("audio", True),
        "lipsync": StreamAvailability(
            "lipsync", False, reason="Needs mouth movement and sound together."
        ),
        "emotion": StreamAvailability(
            "emotion", False, reason="Needs a face and a voice together."
        ),
    },
    UNKNOWN: {},
}

STREAM_LABELS = {
    "visual": "Visual",
    "audio": "Audio",
    "lipsync": "Lip-sync",
    "emotion": "Emotion",
}


def classify(path: str | Path) -> str:
    """The media kind of a path, from its suffix.

    By suffix rather than by decoding. This runs on every rerun to draw the
    routing panel, and opening the file each time would make the page wait on
    disk. A file whose suffix lies is caught later by the decoder, which fails
    loudly, rather than here.
    """
    suffix = Path(path).suffix.lower()
    if suffix in VIDEO_SUFFIXES:
        return VIDEO
    if suffix in IMAGE_SUFFIXES:
        return IMAGE
    if suffix in AUDIO_SUFFIXES:
        return AUDIO
    return UNKNOWN


def availability(kind: str) -> tuple[StreamAvailability, ...]:
    """Every stream and whether this media kind can drive it, in pipeline order."""
    routing = ROUTING.get(kind, {})
    return tuple(routing[name] for name in STREAM_LABELS if name in routing)


def runnable(kind: str) -> tuple[str, ...]:
    """The streams that can run, degraded ones included."""
    return tuple(item.stream for item in availability(kind) if item.available)


def describe(kind: str) -> str:
    """One line naming what will run, for the panel heading."""
    if kind == UNKNOWN:
        return "Unrecognised file type, so nothing can run."
    names = [STREAM_LABELS[s] for s in runnable(kind)]
    if not names:
        return "Nothing can run on this file."
    if len(names) == 1:
        return f"{names[0]} only."
    return ", ".join(names[:-1]) + f" and {names[-1]}."
