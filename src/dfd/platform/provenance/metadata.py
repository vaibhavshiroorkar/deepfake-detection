"""Local media metadata and provenance observations (WS3). Nothing here uploads media or needs a new package.

Four different kinds of statement are kept apart, and each observation says which kind it is:
  ``file_metadata``      what ffprobe reports about the container and streams (ordinary description, not evidence)
  ``heuristic``          a rule-of-thumb flag; descriptive only, many benign causes, never evidence of manipulation on its own
  ``marker_scan``        whether bytes that a C2PA manifest uses (JUMBF boxes labelled "c2pa") occur in the file; presence is NOT
                         verification (signature, certificate chain and hashes are not checked) and absence does not rule out
                         credentials kept elsewhere (a sidecar file or a cloud store)
  ``verification``       a credential that was actually verified; this package cannot produce one (no verifier is installed, see
                         ``c2pa_verifier_available``), so it never appears here
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Optional

KINDS = ("file_metadata", "heuristic", "marker_scan", "verification")
C2PA_MARKERS = (b"jumb", b"c2pa")      # JUMBF superbox type and the C2PA manifest-store label; both must occur (4-byte patterns alone can match by chance)
SCAN_LIMIT = 256 * 1024 * 1024                  # bytes read for the marker scan (upload limit is 512 MB)


def c2pa_verifier_available() -> bool:
    """True only if a real C2PA verifier package can be imported. None is installed in this environment (2026-10-26 check)."""
    return importlib.util.find_spec("c2pa") is not None


def ffprobe(path: str | Path, timeout_s: float = 20.0) -> dict:
    r = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
                       capture_output=True, text=True, timeout=timeout_s, check=False)
    if r.returncode != 0:
        raise ValueError(f"ffprobe could not read the file: {r.stderr.strip()[:200]}")
    return json.loads(r.stdout)


def _obs(kind: str, code: str, text: str, value=None) -> dict:
    assert kind in KINDS
    return {"kind": kind, "code": code, "text": text, "value": value}


def _fps(s: dict) -> Optional[float]:
    r = s.get("avg_frame_rate") or s.get("r_frame_rate") or "0/0"
    try:
        a, b = r.split("/")
        return round(float(a) / float(b), 3) if float(b) else None
    except (ValueError, ZeroDivisionError):
        return None


def describe(info: dict) -> list[dict]:
    """File metadata observations from an ffprobe JSON (ordinary description)."""
    out = []
    f = info.get("format", {})
    out.append(_obs("file_metadata", "CONTAINER", f"container {f.get('format_name')}", f.get("format_name")))
    if f.get("duration"):
        out.append(_obs("file_metadata", "DURATION_S", f"duration {float(f['duration']):.2f} s", round(float(f["duration"]), 3)))
    if f.get("bit_rate"):
        out.append(_obs("file_metadata", "BIT_RATE", f"overall bit rate {int(f['bit_rate'])} bit/s", int(f["bit_rate"])))
    for s in info.get("streams", []):
        t = s.get("codec_type")
        if t == "video":
            out.append(_obs("file_metadata", "VIDEO_STREAM",
                            f"video {s.get('codec_name')} {s.get('width')}x{s.get('height')} at {_fps(s)} fps",
                            {"codec": s.get("codec_name"), "width": s.get("width"), "height": s.get("height"), "fps": _fps(s),
                             "pix_fmt": s.get("pix_fmt")}))
        elif t == "audio":
            out.append(_obs("file_metadata", "AUDIO_STREAM",
                            f"audio {s.get('codec_name')} {s.get('sample_rate')} Hz, {s.get('channels')} channel(s)",
                            {"codec": s.get("codec_name"), "sample_rate": s.get("sample_rate"), "channels": s.get("channels")}))
    tags = {k.lower(): v for k, v in (f.get("tags") or {}).items()}
    for k in ("encoder", "creation_time", "major_brand", "handler_name", "title", "comment"):
        if k in tags:
            out.append(_obs("file_metadata", f"TAG_{k.upper()}", f"container tag {k}: {str(tags[k])[:80]}", str(tags[k])[:80]))
    return out


def heuristics(info: dict) -> list[dict]:
    """Rule-of-thumb flags. Each is descriptive: ordinary editing, transcoding and sharing cause all of them."""
    out = []
    f = info.get("format", {})
    tags = {k.lower(): v for k, v in (f.get("tags") or {}).items()}
    streams = info.get("streams", [])
    if not tags.get("creation_time"):
        out.append(_obs("heuristic", "NO_CREATION_TIME", "no creation time in the container tags; social platforms and editors "
                        "routinely strip it, so this says little"))
    enc = str(tags.get("encoder", ""))
    if enc:
        out.append(_obs("heuristic", "ENCODER_TAG", f"encoder tag '{enc[:60]}': names the last tool that wrote the file, "
                        "not whether content was generated or edited", enc[:60]))
    dur = {s["codec_type"]: float(s["duration"]) for s in streams if s.get("codec_type") in ("video", "audio") and s.get("duration")}
    if "video" in dur and "audio" in dur and abs(dur["video"] - dur["audio"]) > 0.5:
        out.append(_obs("heuristic", "STREAM_DURATION_MISMATCH",
                        f"video stream {dur['video']:.2f} s and audio stream {dur['audio']:.2f} s differ by more than 0.5 s; common "
                        "after trimming or muxing", round(abs(dur["video"] - dur["audio"]), 3)))
    return out


def scan_c2pa_markers(path: str | Path, limit: int = SCAN_LIMIT) -> dict:
    """Search the file bytes for C2PA/JUMBF markers. Returns {scanned_bytes, truncated, found, both_markers}. Not a verification."""
    p = Path(path)
    size = p.stat().st_size
    found = set()
    with open(p, "rb") as fh:
        data = fh.read(limit)
    for m in C2PA_MARKERS:
        if m in data:
            found.add(m.decode())
    return {"scanned_bytes": len(data), "truncated": size > len(data), "found": sorted(found), "both_markers": len(found) == len(C2PA_MARKERS)}


def inspect_provenance(path: str | Path) -> dict:
    """All local observations for one file, plus the honest verification status."""
    info = ffprobe(path)
    scan = scan_c2pa_markers(path)
    obs = describe(info) + heuristics(info)
    if scan["both_markers"]:
        obs.append(_obs("marker_scan", "C2PA_MARKER_PRESENT",
                        f"bytes {scan['found']} occur in the file: a content-credential manifest may be embedded, but nothing was verified",
                        scan["found"]))
    else:
        obs.append(_obs("marker_scan", "C2PA_MARKER_ABSENT",
                        "no embedded C2PA/JUMBF marker bytes found" + (" in the scanned part of the file" if scan["truncated"] else "")
                        + "; credentials kept in a sidecar or a cloud store would not be seen by this scan"))
    return {"observations": obs, "marker_scan": scan, "verifier_available": c2pa_verifier_available(),
            "verified_credentials": []}
