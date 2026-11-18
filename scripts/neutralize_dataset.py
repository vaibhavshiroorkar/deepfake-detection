"""Write shortcut-neutral copies of every clip in a manifest, and a manifest for them.

    uv run python scripts/neutralize_dataset.py --manifest data/manifests/fakeavceleb.csv \
        --dataset FakeAVCeleb --output-root data/neutral --manifest-out data/manifests/fakeavceleb-neutral.csv

The new manifest points at the neutral files. Each row's `sync_start_sec` moves
back by the cut offset, so a LAV-DF window still covers the same forged frames.
`leading_silence_sec` becomes 0 because the silence is gone. A sidecar CSV
keeps every clip's offset and cut lengths, so the change is auditable.
Re-running resumes: finished clips are skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

from deepfake_detection.preprocessing.neutralize import NeutralSettings, neutralize_clip


def _work(job: tuple[str, str, str], settings: NeutralSettings) -> dict[str, object]:
    # Resume is decided by the cut log alone (see main). A video file without a
    # log row may be half-written by a crashed run, so it is always redone.
    clip_id, source, target = job
    return {"clip_id": clip_id, **neutralize_clip(Path(source), Path(target), settings)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("data/neutral"))
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--size", help="one frame size for the corpus, WxH (stretched)")
    parser.add_argument(
        "--video-bitrate",
        help="constant video bitrate such as 160k instead of crf (P-001 A18)",
    )
    arguments = parser.parse_args(argv)
    settings = NeutralSettings(
        size=arguments.size, video_bitrate=arguments.video_bitrate
    )

    with arguments.manifest.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or ())
        rows = list(reader)

    sidecar_path = arguments.manifest_out.with_suffix(".cuts.csv")
    cuts: dict[str, dict] = {}
    if sidecar_path.exists():
        # A refusal for length is final. Any other error (a decoder that could
        # not start under memory pressure) is retried on resume.
        with sidecar_path.open(newline="", encoding="utf-8") as handle:
            cuts = {
                r["clip_id"]: r
                for r in csv.DictReader(handle)
                if not r.get("error") or r["error"].startswith("too short")
            }

    jobs, seen = [], set()
    for row in rows:
        relative = row["video_path"].replace("\\", "/")
        target = arguments.output_root / Path(relative).with_suffix(".mp4")
        row["_target"] = target
        if relative in seen or row["clip_id"] in cuts:
            continue
        seen.add(relative)
        jobs.append((row["clip_id"], str(arguments.data_dir / relative), str(target)))

    cut_fields = (
        "clip_id",
        "offset_sec",
        "kept_sec",
        "cut_leading_sec",
        "cut_trailing_sec",
        "has_audio",
        "error",
    )
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not sidecar_path.exists()
    with sidecar_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cut_fields, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        with ProcessPoolExecutor(arguments.workers) as pool:
            for index, result in enumerate(
                pool.map(partial(_work, settings=settings), jobs, chunksize=8), start=1
            ):
                if not result.get("skipped"):
                    writer.writerow(result)
                    cuts[str(result["clip_id"])] = result
                if index % 500 == 0:
                    handle.flush()
                    print(f"{index}/{len(jobs)}", flush=True)

    # Rows that share a file (LAV-DF windows) share that file's cut.
    by_path = {}
    for row in rows:
        if row["clip_id"] in cuts:
            by_path[str(row["_target"])] = cuts[row["clip_id"]]
    kept, failed = [], 0
    refused: dict[str, dict[str, int]] = {}
    for row in rows:
        cut = cuts.get(row["clip_id"]) or by_path.get(str(row["_target"]))
        if not cut or cut.get("error"):
            failed += 1
            reason = (cut or {}).get("error", "missing") or "missing"
            reason = (
                "too short"
                if str(reason).startswith("too short")
                else "ffmpeg or decode error"
            )
            kind = row.get("manipulation_type", "unknown")
            refused.setdefault(kind, {}).setdefault(reason, 0)
            refused[kind][reason] += 1
            continue
        offset = float(cut["offset_sec"])
        row["video_path"] = (
            str(Path(row["_target"]).relative_to(arguments.data_dir)).replace("\\", "/")
            if Path(row["_target"]).is_relative_to(arguments.data_dir)
            else str(row["_target"])
        )
        row["leading_silence_sec"] = "0.0"
        if row.get("sync_start_sec", "").strip():
            row["sync_start_sec"] = (
                f"{max(0.0, float(row['sync_start_sec']) - offset):.4f}"
            )
        kept.append(row)

    arguments.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with arguments.manifest_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept)
    summary = {
        "dataset": arguments.dataset,
        "source_manifest": str(arguments.manifest),
        "rows_in": len(rows),
        "rows_out": len(kept),
        "rows_failed": failed,
        # A refusal that hits one class harder is a selection effect; report it.
        "refused_by_category": refused,
        "rows_in_by_category": {
            k: sum(1 for r in rows if r.get("manipulation_type") == k)
            for k in sorted({r.get("manipulation_type", "") for r in rows})
        },
        "settings": settings.as_dict(),
    }
    arguments.manifest_out.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if kept else 1


if __name__ == "__main__":
    raise SystemExit(main())
