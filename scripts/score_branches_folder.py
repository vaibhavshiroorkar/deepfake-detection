"""Which branch actually sees the manipulation? Per-branch AUC over a folder.

The fused verdict hides the thing worth knowing. A system can look useless
overall while one branch is carrying real signal, or look adequate while a
single branch does everything and the rest add noise. This scores every clip
through the served engine and reports a separate ROC-AUC per branch, against the
same labels.

The question it exists for: on fully generated media, where video and speech come
from one model, is the generated audio easier to catch than the generated video?
Audio anti-spoofing is a more mature field than video forensics, so the answer
may be yes, and that would move effort from the visual branch to the audio one.

Layout is the same as `score_media_folder.py`: `real/` and `fake/<source>/`.
Clips a branch could not read are excluded from that branch's AUC and reported
as coverage, so a branch is never credited for clips it abstained on.

    uv run python scripts/score_branches_folder.py --folder data/df26-scored
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    from sklearn.metrics import roc_auc_score

    from deepfake_detection.dashboard.configuration import dashboard_defaults
    from deepfake_detection.dashboard.lib import media_kind
    from deepfake_detection.inference.multimodal import load_multimodal_engine

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Clips per group, 0 for all. Useful for a first pass.",
    )
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args(argv)

    files: list[tuple[Path, str, int]] = []
    real_root = arguments.folder / "real"
    if real_root.is_dir():
        found = [
            p
            for p in sorted(real_root.rglob("*"))
            if p.is_file() and media_kind.classify(p) != media_kind.UNKNOWN
        ]
        files += [(p, "real", 0) for p in (found[: arguments.limit] if arguments.limit else found)]
    fake_root = arguments.folder / "fake"
    if fake_root.is_dir():
        for source in sorted(p for p in fake_root.iterdir() if p.is_dir()):
            found = [
                p
                for p in sorted(source.rglob("*"))
                if p.is_file() and media_kind.classify(p) != media_kind.UNKNOWN
            ]
            chosen = found[: arguments.limit] if arguments.limit else found
            files += [(p, source.name, 1) for p in chosen]
    if not files:
        print(f"No media under {arguments.folder}")
        return 1
    print(f"{len(files)} clips\n", flush=True)

    root = Path.cwd()
    defaults = dashboard_defaults(root=root)
    engine = load_multimodal_engine(
        run_dir=defaults.stream_run,
        code_version=defaults.stream_code_version,
        fusion_path=defaults.gate_fusion_checkpoint,
        device=arguments.device,
        root=root,
    )

    # branch -> list of (logit, label, group)
    per_branch: dict[str, list[tuple[float, int, str]]] = defaultdict(list)
    fused: list[tuple[float, int, str]] = []
    failures = 0
    for index, (path, group, label) in enumerate(files, start=1):
        try:
            result = engine.predict(path)
        except (OSError, ValueError, RuntimeError) as error:
            failures += 1
            print(f"  {path.name[:44]:44} failed {type(error).__name__}: {error}")
            continue
        for branch, logit in result.branch_logits.items():
            per_branch[branch].append((float(logit), label, group))
        if result.probability is not None:
            fused.append((float(result.probability), label, group))
        if index % 50 == 0:
            print(f"  {index} of {len(files)}", flush=True)

    def auc_of(items) -> float | None:
        labels = [label for _, label, _ in items]
        if len(set(labels)) != 2:
            return None
        return float(roc_auc_score(labels, [value for value, _, _ in items]))

    print(f"\n{'branch':26} {'ROC-AUC':>8}  {'clips':>6}  coverage")
    print("-" * 58)
    summary: dict[str, dict] = {}
    for branch, items in sorted(per_branch.items()):
        value = auc_of(items)
        summary[branch] = {
            "roc_auc": value,
            "clips": len(items),
            "coverage": len(items) / len(files),
        }
        shown = f"{value:.4f}" if value is not None else "one class"
        print(
            f"{branch:26} {shown:>8}  {len(items):6}  "
            f"{len(items) / len(files):.1%}"
        )
    value = auc_of(fused)
    summary["fused"] = {
        "roc_auc": value,
        "clips": len(fused),
        "coverage": len(fused) / len(files),
    }
    shown = f"{value:.4f}" if value is not None else "one class"
    print(f"{'fused':26} {shown:>8}  {len(fused):6}  {len(fused) / len(files):.1%}")

    # Per branch and per generated source, since one aggregate hides the
    # per-generator collapse that is the norm in this problem.
    groups = sorted({group for _, _, group in fused} - {"real"})
    if groups:
        print("\nper source, branch by branch")
        header = f"{'branch':26}" + "".join(f"{g[:12]:>14}" for g in groups)
        print(header)
        print("-" * len(header))
        for branch, items in sorted(per_branch.items()):
            cells = ""
            for group in groups:
                subset = [i for i in items if i[2] in (group, "real")]
                value = auc_of(subset)
                cells += f"{(f'{value:.3f}' if value is not None else 'n/a'):>14}"
                summary.setdefault(branch, {}).setdefault("per_source", {})[group] = value
            print(f"{branch:26}{cells}")

    if failures:
        print(f"\n{failures} clips failed entirely")
    output = arguments.output or arguments.folder / "branch-scores.json"
    output.write_text(
        json.dumps({"branches": summary, "clips": len(files)}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
