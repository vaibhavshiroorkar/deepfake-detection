"""Score a folder of media through the served engine, grouped by source.

Built for the case the current system fails hardest at: clips from Veo, Sora,
Higgsfield and the rest, which it detects at 0 of 10 on the one such family
measured. A number on your own generators is the baseline every later
improvement gets compared against, and this is the shortest path to it.

No manifest and no cache. Each file goes through the same preprocessing and the
same engine the dashboard uses, so the numbers here are what the deployed system
would actually say.

Layout, by directory name:

    folder/
      real/                 authentic clips
      fake/<generator>/     generated or manipulated clips, one directory each

The generator directory name becomes the group label. Detection rates carry
Wilson intervals because per-generator counts are small, and an interval on 0 of
10 reaches 28 percent, which a bare fraction hides.

The distribution gate runs alongside and is reported separately. It is not a
detector. It says whether the clip's motion statistics sit inside the range the
training corpus covered, which is the honest answer for media the system was
never trained on.

    uv run python scripts/score_media_folder.py --folder data/my-generators
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def _wilson(successes: int, total: int) -> tuple[float, float]:
    """95 percent Wilson interval, because these counts are small."""
    if total == 0:
        return (0.0, 1.0)
    z = 1.96
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = (
        z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / denominator
    )
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _walk(folder: Path):
    """Every media file under the folder, with its group and its true label."""
    from deepfake_detection.dashboard.lib import media_kind

    real_root = folder / "real"
    if real_root.is_dir():
        for path in sorted(real_root.rglob("*")):
            if path.is_file() and media_kind.classify(path) != media_kind.UNKNOWN:
                yield path, "real", 0

    fake_root = folder / "fake"
    if fake_root.is_dir():
        for generator in sorted(p for p in fake_root.iterdir() if p.is_dir()):
            for path in sorted(generator.rglob("*")):
                if path.is_file() and media_kind.classify(path) != media_kind.UNKNOWN:
                    yield path, generator.name, 1


def main(argv: list[str] | None = None) -> int:
    from deepfake_detection.dashboard.configuration import dashboard_defaults
    from deepfake_detection.inference.multimodal import load_multimodal_engine

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Overrides the per-media-kind threshold in the head.")
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args(argv)

    files = list(_walk(arguments.folder))
    if not files:
        print(
            f"No media under {arguments.folder}. Expected real/ and "
            "fake/<generator>/ directories."
        )
        return 1
    print(f"{len(files)} files under {arguments.folder}\n")

    root = Path.cwd()
    defaults = dashboard_defaults(root=root)
    engine = load_multimodal_engine(
        run_dir=defaults.stream_run,
        code_version=defaults.stream_code_version,
        fusion_path=defaults.gate_fusion_checkpoint,
        device=arguments.device,
        root=root,
    )

    rows = []
    groups: dict[str, list[dict]] = defaultdict(list)
    for path, group, label in files:
        try:
            result = engine.predict(path)
        except (OSError, ValueError, RuntimeError) as error:
            print(f"  {path.name}: failed, {type(error).__name__}: {error}")
            continue

        # The gate is read from the blockers the engine already attached, so the
        # clip is not preprocessed a second time.
        outside = any(
            blocker.startswith("outside_training_distribution")
            for blocker in result.blockers
        )

        row = {
            "file": str(path.relative_to(arguments.folder)).replace("\\", "/"),
            "group": group,
            "label": label,
            "media_kind": result.media_kind,
            "verdict": result.verdict,
            "probability": result.probability,
            "threshold": result.threshold,
            "outside_training_distribution": outside,
            "blockers": list(result.blockers),
        }
        rows.append(row)
        groups[group].append(row)
        print(
            f"  {row['file'][:52]:52} {result.verdict:13} "
            + (
                f"{result.probability:.3f}"
                if result.probability is not None
                else "  n/a"
            )
            + ("  outside distribution" if outside else "")
        )

    print(f"\n{'group':24} {'called fake':>12} {'clips':>6}  95% interval  outside")
    print("-" * 74)
    summary: dict[str, dict] = {}
    for group, items in sorted(groups.items()):
        called = sum(1 for item in items if item["verdict"] == "fake")
        outside_count = sum(1 for item in items if item["outside_training_distribution"])
        low, high = _wilson(called, len(items))
        summary[group] = {
            "clips": len(items),
            "called_fake": called,
            "rate": called / len(items),
            "ci": [low, high],
            "outside_distribution": outside_count,
            "label": items[0]["label"],
        }
        print(
            f"{group:24} {called:7} / {len(items):<4} {len(items):6}  "
            f"[{low:.0%}, {high:.0%}]   {outside_count}"
        )

    # A ranking metric only when both classes are present, which needs real
    # clips in the folder. Without them the table above is a detection rate and
    # says nothing about false alarms.
    scored = [r for r in rows if r["probability"] is not None]
    labels = {r["label"] for r in scored}
    auc = None
    if len(labels) == 2:
        from sklearn.metrics import roc_auc_score

        auc = float(
            roc_auc_score(
                [r["label"] for r in scored], [r["probability"] for r in scored]
            )
        )
        print(f"\nROC-AUC over {len(scored)} scored clips: {auc:.4f}")
    else:
        print(
            "\nOne class only, so no ranking metric. Add real clips under real/ "
            "to get an AUC and a false-alarm rate."
        )

    output = arguments.output or arguments.folder / "scores.json"
    output.write_text(
        json.dumps(
            {"groups": summary, "roc_auc": auc, "clips": rows},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
