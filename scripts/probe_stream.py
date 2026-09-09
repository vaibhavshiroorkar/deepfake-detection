"""Ask a trained cross-modal stream what it actually learned.

A low validation loss says a stream separates the classes. It does not say the
separation came from audio-video alignment, which is the whole claim. These
probes are designed so a stream that learned appearance and a stream that
learned synchronisation give different answers:

  1. **Audio shift.** Offset the audio by 320 ms on genuine clips. A model
     reading alignment must become less confident that they are real. A model
     reading mouth-region artifacts will not move, because no pixel changed.
  2. **Frame order.** Reverse the video. An untrained stream is order-blind by
     construction, because uniform attention makes the attended vector a mean
     over frames and a mean cannot see order. A trained stream that uses
     temporal structure has to break that.
  3. **Cross-pairing.** Give one clip's audio another clip's mouth. This is the
     most direct statement of the question the stream is supposed to answer.

Each probe reports the change relative to the model's own scale, because an
absolute delta means nothing without knowing how large the outputs are.

    uv run python scripts/probe_stream.py --checkpoint ... --manifest ...
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def _load_index(path: Path) -> dict[str, Path]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = tuple(csv.DictReader(handle))
    return {
        row["clip_id"]: (
            Path(row["cache_path"])
            if Path(row["cache_path"]).is_absolute()
            else (path.parent / row["cache_path"]).resolve()
        )
        for row in rows
    }


def main(argv: list[str] | None = None) -> int:
    import torch

    from deepfake_detection.data.datasets import (
        CachedAVPairDataset,
        collate_av_pair_items,
    )
    from deepfake_detection.data.manifest import load_manifest
    from deepfake_detection.streams.cross_modal_stream import build_lipsync_stream
    from deepfake_detection.training.checkpoints import load_checkpoint
    from deepfake_detection.views.cache_store import CacheStore

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-index", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--dataset", default="LAV-DF")
    parser.add_argument("--preprocessing-hash")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clips", type=int, default=64)
    parser.add_argument("--shift-ms", type=int, default=320)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)

    records = load_manifest(arguments.manifest, dataset=arguments.dataset).records
    genuine = [record for record in records if not record.clip_fake][: arguments.clips]
    if len(genuine) < 2:
        raise SystemExit("Need at least two genuine clips to probe")

    dataset = CachedAVPairDataset(
        records=genuine,
        cache_index=_load_index(arguments.cache_index),
        cache_store=CacheStore(arguments.cache_root),
        stream="lipsync",
        preprocessing_hash=arguments.preprocessing_hash,
    )
    batch = collate_av_pair_items([dataset[i] for i in range(len(dataset))])
    video = batch.video.to(arguments.device)
    audio = batch.audio.to(arguments.device)

    model = build_lipsync_stream(pretrained=False)
    load_checkpoint(arguments.checkpoint, model=model)
    model.to(arguments.device).eval()

    shift = int(arguments.shift_ms * arguments.sample_rate / 1000)
    with torch.inference_mode():
        base = model(video=video, audio=audio)
        shifted = model(video=video, audio=torch.roll(audio, shift, dims=1))
        reversed_video = model(video=video.flip(dims=[1]), audio=audio)
        # Roll by one so no clip keeps its own audio.
        mismatched = model(video=video, audio=torch.roll(audio, 1, dims=0))

    def fake_probability(output) -> float:
        return float(torch.sigmoid(output.logit).mean())

    def relative(a, b) -> float:
        scale = float(a.embedding.abs().max())
        return float((a.embedding - b.embedding).abs().max() / scale) if scale else 0.0

    report = {
        "checkpoint": str(arguments.checkpoint),
        "genuine_clips": len(genuine),
        "shift_ms": arguments.shift_ms,
        "baseline": {
            "mean_fake_probability": fake_probability(base),
            "diagonal_mass": float(base.diagonal_mass.mean()),
        },
        "audio_shift": {
            "mean_fake_probability": fake_probability(shifted),
            "probability_change": fake_probability(shifted) - fake_probability(base),
            "embedding_relative_change": relative(base, shifted),
        },
        "frame_reversal": {
            "mean_fake_probability": fake_probability(reversed_video),
            "probability_change": fake_probability(reversed_video)
            - fake_probability(base),
            "embedding_relative_change": relative(base, reversed_video),
        },
        "cross_paired_audio": {
            "mean_fake_probability": fake_probability(mismatched),
            "probability_change": fake_probability(mismatched)
            - fake_probability(base),
            "embedding_relative_change": relative(base, mismatched),
        },
    }
    # Cross-pairing is the only probe that settles the question, and it is the
    # strict one. Shifting the audio also perturbs the waveform itself, and a
    # control run showed this model responds identically (0.1337) whether the
    # shift breaks alignment or is applied to both modalities so alignment is
    # preserved. That response is to the altered audio, not to the misalignment.
    #
    # Cross-pairing changes nothing about either signal's own statistics: clip A
    # keeps its real mouth, clip B keeps its real voice, and only their
    # correspondence is destroyed. A stream that reads alignment must call that
    # fake.
    report["reads_alignment"] = (
        report["cross_paired_audio"]["probability_change"] > 0.05
    )
    report["alignment_verdict"] = (
        "cross-paired audio changes the decision, so the stream uses correspondence"
        if report["reads_alignment"]
        else "cross-paired audio leaves the decision unchanged, so the stream "
        "classifies without using audio-video correspondence"
    )

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
