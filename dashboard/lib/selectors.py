"""Shared Dataset/Target/Clip selection for the preprocessing pages.

Pure filtering (filter_manifest) is separated from the Streamlit widgets
(render_selection) so the filter logic is unit-testable without a running app.
All shared st.session_state keys are owned here.
"""
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DATA_DIR = _REPO_ROOT / "data"

# Label -> manifest filename. Deepfake-Eval-2024 is listed but has no manifest
# yet, so render_selection() disables it until data/deepfake_eval.csv exists.
DATASETS = {
    "train": "train.csv",
    "val": "val.csv",
    "test": "test.csv",
    "full_manifest": "full_manifest.csv",
}
_FUTURE_DATASETS = {"deepfake_eval": "deepfake_eval.csv"}

MANIP_TYPES = ["RealVideo-RealAudio", "RealVideo-FakeAudio",
               "FakeVideo-RealAudio", "FakeVideo-FakeAudio"]


def load_manifest(dataset: str) -> pd.DataFrame:
    fname = DATASETS.get(dataset) or _FUTURE_DATASETS.get(dataset)
    if fname is None:
        raise KeyError(f"Unknown dataset {dataset!r}")
    path = DATA_DIR / fname
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run the preprocessing pipeline first.")
    return pd.read_csv(path)


def filter_manifest(df: pd.DataFrame, manip_types: list[str],
                    methods: list[str], label_filter: str) -> pd.DataFrame:
    out = df
    if manip_types:
        out = out[out["manipulation_type"].isin(manip_types)]
    if methods:
        out = out[out["method"].isin(methods)]
    if label_filter == "real":
        out = out[out["label"] == 0]
    elif label_filter == "fake":
        out = out[out["label"] == 1]
    return out.reset_index(drop=True)


def render_selection():
    """Render shared Dataset/Target/Clip controls; return the selected clip row."""
    import streamlit as st

    st.subheader("Selection")
    c1, c2 = st.columns(2)
    with c1:
        options = list(DATASETS.keys()) + list(_FUTURE_DATASETS.keys())

        def _fmt(name):
            return f"{name} (not available)" if name in _FUTURE_DATASETS else name

        dataset = st.selectbox("Dataset", options, format_func=_fmt, key="sel_dataset")
        if dataset in _FUTURE_DATASETS:
            st.info(f"'{dataset}' has no manifest yet. Pick a built split.")
            return None
    with c2:
        sample_n = st.slider("Sample size", 5, 40, 15, key="sel_sample_n")
        seed = st.number_input("Seed", value=42, step=1, key="sel_seed")

    df = load_manifest(dataset)
    methods_present = sorted(df["method"].dropna().unique().tolist())
    t1, t2, t3 = st.columns(3)
    with t1:
        manip = st.multiselect("Target: manipulation type", MANIP_TYPES, key="sel_types")
    with t2:
        methods = st.multiselect("Target: method", methods_present, key="sel_methods")
    with t3:
        label_filter = st.radio("Target: label", ["all", "real", "fake"], key="sel_label")

    filtered = filter_manifest(df, manip, methods, label_filter)
    if len(filtered) == 0:
        st.warning("No clips match the current target filters.")
        return None

    sample = filtered.sample(n=min(int(sample_n), len(filtered)),
                             random_state=int(seed)).reset_index(drop=True)

    def _label(i):
        r = sample.iloc[i]
        tag = "REAL" if r["label"] == 0 else "fake"
        return f"[{tag}] {r['manipulation_type']} — {r['clip_id']}"

    idx = st.selectbox("Clip", range(len(sample)), format_func=_label, key="sel_clip_idx")
    return sample.iloc[idx]
