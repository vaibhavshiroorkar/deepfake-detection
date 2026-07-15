"""
Stage 1 - independent leakage verification.

Re-checks the WRITTEN split CSVs from scratch (not trusting build_splits.py's
own assertion): confirms no source identity, and no physical video_path,
appears in more than one split. This is the Research-workstream sign-off from
docs/stage-1-plan.md -- an independent pair of eyes on the #1 project risk.
"""
import sys
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


def verify():
    splits = {}
    for name in ["train", "val", "test"]:
        p = DATA_DIR / f"{name}.csv"
        if not p.exists():
            raise FileNotFoundError(f"{p} not found -- run build_splits.py first.")
        splits[name] = pd.read_csv(p)

    # 1. Identity disjointness.
    id_sets = {name: set(df["source"]) for name, df in splits.items()}
    overlaps = {
        f"{a}&{b}": id_sets[a] & id_sets[b]
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]
    }
    any_id_leak = any(len(v) > 0 for v in overlaps.values())
    for k, v in overlaps.items():
        print(f"identity overlap {k}: {len(v)}" + (f"  LEAK: {sorted(v)[:5]}" if v else ""))

    # 2. Physical-file disjointness (belt and suspenders).
    path_sets = {name: set(df["video_path"]) for name, df in splits.items()}
    any_path_leak = any(
        len(path_sets[a] & path_sets[b]) > 0
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]
    )

    # 3. Random named spot-check: print which split a few identities live in.
    import random
    rng = random.Random(0)
    all_ids = sorted(set().union(*id_sets.values()))
    for ident in rng.sample(all_ids, k=min(5, len(all_ids))):
        homes = [name for name in splits if ident in id_sets[name]]
        print(f"identity {ident} -> {homes}")

    if any_id_leak or any_path_leak:
        raise RuntimeError("LEAKAGE DETECTED -- do not proceed to training.")
    print("\nPASS: splits are identity-disjoint and file-disjoint. Signed off.")


if __name__ == "__main__":
    verify()
