"""Shared Dataset -> Split -> Target -> Clip selection for the preprocessing page.

Selection is two-level: pick a DATASET (FakeAVCeleb, Deepfake-Eval-2024,
FaceForensics++, Celeb-DF), then a SPLIT within it. Only FakeAVCeleb has
manifests today; the others render as "not available" until their split CSVs
exist on disk (see DATASETS for where each is expected).

Pure filtering (filter_manifest) is separated from the Streamlit widgets
(render_selection) so the filter logic stays unit-testable without a running
app. All shared st.session_state keys are owned here.
"""
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DATA_DIR = _REPO_ROOT / "data"

# dataset -> {split_name: manifest path relative to data/}. FakeAVCeleb's
# manifests are flat in data/ (written by the preprocessing pipeline). The other
# datasets are declared here with their expected locations but have no manifests
# yet, so they surface as "not available". Deepfake-Eval-2024 is a held-out eval
# set (PROJECT_OVERVIEW.md §5), so it has only a test split.
DATASETS = {
    "FakeAVCeleb": {
        "train": "train.csv",
        "val": "val.csv",
        "test": "test.csv",
        "full_manifest": "full_manifest.csv",
    },
    "Deepfake-Eval-2024": {
        "test": "deepfake_eval/test.csv",
    },
    "FaceForensics++": {
        "train": "faceforensics/train.csv",
        "val": "faceforensics/val.csv",
        "test": "faceforensics/test.csv",
    },
    "Celeb-DF": {
        "train": "celebdf/train.csv",
        "val": "celebdf/val.csv",
        "test": "celebdf/test.csv",
    },
}

MANIP_TYPES = ["RealVideo-RealAudio", "RealVideo-FakeAudio",
               "FakeVideo-RealAudio", "FakeVideo-FakeAudio"]


def available_splits(dataset: str) -> list[str]:
    """Splits of a dataset whose manifest file actually exists on disk."""
    splits = DATASETS.get(dataset, {})
    return [name for name, rel in splits.items() if (DATA_DIR / rel).exists()]


def load_manifest(dataset: str, split: str) -> pd.DataFrame:
    rel = DATASETS.get(dataset, {}).get(split)
    if rel is None:
        raise KeyError(f"Unknown dataset/split {dataset!r}/{split!r}")
    path = DATA_DIR / rel
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — this split has no manifest yet.")
    return pd.read_csv(path)


def filter_manifest(df: pd.DataFrame, manip_types: list[str],
                    methods: list[str], label_filter: str) -> pd.DataFrame:
    out = df
    if manip_types and "manipulation_type" in out.columns:
        out = out[out["manipulation_type"].isin(manip_types)]
    if methods and "method" in out.columns:
        out = out[out["method"].isin(methods)]
    if label_filter == "real":
        out = out[out["label"] == 0]
    elif label_filter == "fake":
        out = out[out["label"] == 1]
    return out.reset_index(drop=True)


def render_selection():
    """Render shared Dataset/Split/Target/Clip controls; return the selected clip row."""
    import streamlit as st

    st.header("Selection")
    c1, c2, c3 = st.columns(3)

    with c1:
        def _fmt_dataset(name):
            return name if available_splits(name) else f"{name} (not available)"

        dataset = st.selectbox("Dataset", list(DATASETS.keys()),
                               format_func=_fmt_dataset, key="sel_dataset")

    splits = available_splits(dataset)
    if not splits:
        st.info(f"'{dataset}' has no manifests yet. Only FakeAVCeleb is built so far.")
        return None

    with c2:
        split = st.selectbox("Split", splits, key="sel_split")
    with c3:
        sample_n = st.slider("Sample size", 5, 40, 15, key="sel_sample_n")

    df = load_manifest(dataset, split)

    # Target filters — only shown for columns this dataset actually has.
    has_type = "manipulation_type" in df.columns
    has_method = "method" in df.columns
    t1, t2, t3 = st.columns(3)
    manip, methods = [], []
    with t1:
        if has_type:
            manip = st.multiselect("Target: manipulation type", MANIP_TYPES, key="sel_types")
    with t2:
        if has_method:
            methods_present = sorted(df["method"].dropna().unique().tolist())
            methods = st.multiselect("Target: method", methods_present, key="sel_methods")
    with t3:
        label_filter = st.radio("Target: label", ["all", "real", "fake"], key="sel_label")

    seed = st.number_input("Sample seed", value=42, step=1, key="sel_seed")

    filtered = filter_manifest(df, manip, methods, label_filter)
    if len(filtered) == 0:
        st.warning("No clips match the current target filters.")
        return None

    sample = filtered.sample(n=min(int(sample_n), len(filtered)),
                             random_state=int(seed)).reset_index(drop=True)

    def _label(i):
        r = sample.iloc[i]
        tag = "REAL" if r["label"] == 0 else "fake"
        mtype = r["manipulation_type"] if has_type else ""
        return f"[{tag}] {mtype} — {r['clip_id']}".replace("—  —", "—")

    idx = st.selectbox("Clip", range(len(sample)), format_func=_label, key="sel_clip_idx")
    return sample.iloc[idx]
