"""
Stage 1 - identity-disjoint train/val/test splits.

Splitting unit = the `source` identity (500 VoxCeleb base identities). Every
clip is assigned to a split by its source id, so all clips sharing the same
underlying real footage stay together in one split. This blocks the dominant
FakeAVCeleb leakage mode: a real clip in train and its fake derivative in test
share background/lighting/framing, which a model can exploit for a fake-high
AUC. See audit_dataset.py's header and docs/stage-1-plan.md for why a
connected-components split over swap pairs is impossible here (the swap graph
is one fully-connected component of 578 identities).

Balancing: FakeAVCeleb is 500 real vs ~21k fake. This script undersamples
fakes per split to REAL_TO_FAKE_RATIO so val/test aren't ~98% fake (which
makes precision/recall/F1 unstable). Train-time class balancing
(WeightedRandomSampler, Stage 2) still happens on top of this.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
FULL_MANIFEST = REPO_ROOT / "data" / "full_manifest.csv"
OUT_DIR = REPO_ROOT / "data"

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED = 42
IDENTITY_COL = "source"  # the split unit
# Keep real:fake at 1:3 per split -- more fake examples (more manipulation
# types to learn) without making real examples vanishingly rare.
REAL_TO_FAKE_RATIO = 1 / 3


def assign_identity_splits(identities: list[str], rng: np.random.Generator) -> dict[str, str]:
    identities = sorted(identities)  # deterministic order before shuffling
    shuffled = rng.permutation(identities)
    n = len(shuffled)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])
    split_of = {}
    for i, ident in enumerate(shuffled):
        if i < n_train:
            split_of[ident] = "train"
        elif i < n_train + n_val:
            split_of[ident] = "val"
        else:
            split_of[ident] = "test"
    return split_of


def undersample_fakes(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    real = df[df["label"] == 0]
    fake = df[df["label"] == 1]
    if len(real) == 0:
        return df
    max_fake = int(len(real) / REAL_TO_FAKE_RATIO)
    if len(fake) > max_fake:
        fake = fake.sample(n=max_fake, random_state=int(rng.integers(0, 1_000_000)))
    return pd.concat([real, fake], ignore_index=True)


def build_splits():
    if not FULL_MANIFEST.exists():
        raise FileNotFoundError(f"{FULL_MANIFEST} not found -- run audit_dataset.py first.")

    df = pd.read_csv(FULL_MANIFEST)
    rng = np.random.default_rng(RANDOM_SEED)

    identities = df[IDENTITY_COL].unique().tolist()
    split_of = assign_identity_splits(identities, rng)
    df["split"] = df[IDENTITY_COL].map(split_of)

    # Leakage guard: assert no source identity spans more than one split.
    spanning = df.groupby(IDENTITY_COL)["split"].nunique()
    leaked = spanning[spanning > 1]
    if len(leaked) > 0:
        raise RuntimeError(f"Identity leakage: {leaked.index.tolist()}")
    print(f"Leakage check passed: all {len(identities)} source identities live in exactly one split.")

    for split_name in ["train", "val", "test"]:
        split_df = df[df["split"] == split_name].copy()
        split_df = undersample_fakes(split_df, rng)
        # shuffle rows so real/fake aren't blocked together on disk
        split_df = split_df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
        out_path = OUT_DIR / f"{split_name}.csv"
        split_df.to_csv(out_path, index=False)

        n_real = int((split_df["label"] == 0).sum())
        n_fake = int((split_df["label"] == 1).sum())
        print(
            f"{split_name}: {len(split_df)} clips (real={n_real}, fake={n_fake}, "
            f"1:{n_fake / max(n_real, 1):.1f}), {split_df[IDENTITY_COL].nunique()} identities -> {out_path.name}"
        )


if __name__ == "__main__":
    build_splits()
