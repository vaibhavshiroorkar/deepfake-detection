"""Turn LAV-DF into manifest rows, cutting a matched pair from every forgery.

LAV-DF hides a short content-driven forgery inside an otherwise genuine clip: a
word is chosen for the largest swing it causes in perceived sentiment,
text-to-speech generates replacement audio, and facial reenactment follows it.
The manipulated span is 0.66 seconds at the median inside a 7.3 second clip, and
`metadata.json` says exactly where it sits.

That annotation is what makes this adapter different from the others. Instead of
one row per clip it emits **two rows per forgery**:

  - a window centred on the manipulated span, labelled fake
  - the largest window that misses every manipulated span, labelled real

Both come from the same file. Same speaker, same lighting, same microphone, same
codec, same re-encoding pass. They differ in one thing: whether that particular
two seconds was manipulated. A model cannot separate them by recognising the
generator's compression signature, because both windows carry it.

That matters because the measured failure this project is chasing is exactly
that kind of shortcut. A visual model trained on FakeAVCeleb reached 1.0000
ROC-AUC in-domain and then called 130 of 155 genuine Celeb-DF videos fake: it
had learned what its training corpus looked like. A first lip-sync run on
FakeAVCeleb showed the same thing from the other side, with attention mass
stuck at chance for every epoch, because FakeAVCeleb manipulates whole clips and
so never poses the question of *when* the audio stopped matching the mouth.

99.7 percent of LAV-DF forgeries leave room for both windows.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

DATASET = "LAV-DF"

# The synchronisation window the cache cuts, from views/timeline.py ViewConfig.
SYNC_SECONDS = 2.0

MANIFEST_COLUMNS = [
    "clip_id",
    "dataset",
    "video_path",
    "manipulation_type",
    "method",
    "source",
    "target1",
    "target2",
    "race",
    "gender",
    "sync_start_sec",
]


def manipulation_type(*, modify_video: bool, modify_audio: bool) -> str:
    """LAV-DF's two flags map exactly onto the project's four-value enum.

    Unlike Celeb-DF-v2 and MNW, which have to be forced into
    `FakeVideo-RealAudio` because they carry no audio manipulation at all, this
    corpus populates every one of the four honestly, including the
    `RealVideo-FakeAudio` case that FakeAVCeleb has only 500 examples of.
    """
    if modify_video and modify_audio:
        return "FakeVideo-FakeAudio"
    if modify_video:
        return "FakeVideo-RealAudio"
    if modify_audio:
        return "RealVideo-FakeAudio"
    return "RealVideo-RealAudio"


def forged_window(periods: list[list[float]], duration: float) -> float:
    """Window start that centres the longest manipulated span.

    Clamped to the clip, so a forgery in the final second still yields a window
    that fits rather than one running off the end.
    """
    start, end = max(periods, key=lambda span: span[1] - span[0])
    centre = (start + end) / 2
    return max(0.0, min(centre - SYNC_SECONDS / 2, duration - SYNC_SECONDS))


def genuine_window(periods: list[list[float]], duration: float) -> float | None:
    """Start of the widest span that touches no manipulated period.

    None when the forgeries leave no room, which is 0.3 percent of clips. Those
    contribute their fake window only; silently pairing them against overlapping
    audio would put a manipulated span under a real label.
    """
    gaps: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in sorted(periods):
        if start - cursor >= SYNC_SECONDS:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if duration - cursor >= SYNC_SECONDS:
        gaps.append((cursor, duration))
    if not gaps:
        return None
    widest = max(gaps, key=lambda gap: gap[1] - gap[0])
    return widest[0]


def _row(
    *,
    clip_id: str,
    video_path: str,
    kind: str,
    method: str,
    source: str,
    sync_start: float,
) -> dict[str, object]:
    return {
        "clip_id": clip_id,
        "dataset": DATASET,
        "video_path": video_path,
        "manipulation_type": kind,
        "method": method,
        "source": source,
        "target1": "",
        "target2": "",
        # LAV-DF publishes no demographic metadata.
        "race": "unknown",
        "gender": "unknown",
        "sync_start_sec": round(sync_start, 3),
    }


def manifest_from_lavdf(
    root: Path,
    data_dir: Path,
    *,
    split: str | None = None,
    matched_pairs: bool = True,
) -> pd.DataFrame:
    """Manifest rows for a LAV-DF checkout.

    `root` holds `metadata.min.json` and the train/dev/test directories.
    `data_dir` is the repo's data/ directory, which video_path is relative to.
    With `split`, only that partition is emitted, honouring LAV-DF's own
    protocol rather than inventing one.
    """
    root, data_dir = Path(root), Path(data_dir)
    metadata = json.loads((root / "metadata.min.json").read_text(encoding="utf-8"))

    rows: list[dict[str, object]] = []
    for entry in metadata:
        if split is not None and entry.get("split") != split:
            continue
        relative = root / entry["file"]
        if not relative.is_file():
            continue
        video_path = str(relative.relative_to(data_dir))
        stem = Path(entry["file"]).stem
        periods = entry.get("fake_periods") or []
        duration = float(entry["duration"])
        if duration < SYNC_SECONDS:
            continue

        if not periods:
            rows.append(
                _row(
                    clip_id=f"lavdf__{stem}",
                    video_path=video_path,
                    kind="RealVideo-RealAudio",
                    method="real",
                    # A genuine clip is its own identity group.
                    source=f"lavdf-{stem}",
                    sync_start=0.0,
                )
            )
            continue

        # A forgery and the real clip it was made from share an identity group,
        # so `build_source_split` cannot place them in different partitions and
        # let the model see the same face on both sides of the split.
        origin = Path(entry.get("original") or entry["file"]).stem
        source = f"lavdf-{origin}"
        rows.append(
            _row(
                clip_id=f"lavdf__{stem}__forged",
                video_path=video_path,
                kind=manipulation_type(
                    modify_video=bool(entry.get("modify_video")),
                    modify_audio=bool(entry.get("modify_audio")),
                ),
                method="lavdf-forged",
                source=source,
                sync_start=forged_window(periods, duration),
            )
        )
        if not matched_pairs:
            continue
        genuine = genuine_window(periods, duration)
        if genuine is None:
            continue
        rows.append(
            _row(
                clip_id=f"lavdf__{stem}__genuine",
                video_path=video_path,
                # An untouched span of a manipulated file. The re-encoding is
                # present here too, which is the whole point: it denies the
                # model the codec shortcut.
                kind="RealVideo-RealAudio",
                method="lavdf-genuine-span",
                source=source,
                sync_start=genuine,
            )
        )

    if not rows:
        raise ValueError(f"No LAV-DF clips found under {root}")
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
