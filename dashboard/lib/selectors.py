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


def search_manifest(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """Substring search over clip_id and (if present) the source identity."""
    q = query.strip().lower()
    if not q:
        return df
    cols = ["clip_id"] + (["source"] if "source" in df.columns else [])
    mask = None
    for c in cols:
        hit = df[c].astype(str).str.lower().str.contains(q, regex=False)
        mask = hit if mask is None else (mask | hit)
    return df[mask]


def group_by_identity(df: pd.DataFrame) -> pd.DataFrame:
    """Sort so all variants of one identity sit together (source, then type)."""
    group_col = "source" if "source" in df.columns else "clip_id"
    sort_cols = [group_col] + (["manipulation_type"] if "manipulation_type" in df.columns else [])
    return df.sort_values(sort_cols).reset_index(drop=True)


def render_selection():
    """Render Dataset/Split + a modal Clip picker; return the selected clip row."""
    import streamlit as st

    st.header("Selection")
    c1, c2 = st.columns(2)
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

    df = load_manifest(dataset, split)

    # Reset the chosen clip whenever the dataset/split changes.
    ctx = f"{dataset}/{split}"
    if st.session_state.get("sel_context") != ctx:
        st.session_state["sel_context"] = ctx
        st.session_state.pop("sel_clip_id", None)
        st.session_state["show_picker"] = False

    # Default to the first clip so the page isn't empty on first load.
    sel_id = st.session_state.get("sel_clip_id")
    if sel_id is None or not (df["clip_id"] == sel_id).any():
        sel_id = df.iloc[0]["clip_id"]
        st.session_state["sel_clip_id"] = sel_id

    crow = df[df["clip_id"] == sel_id].iloc[0]
    b1, b2 = st.columns([3, 1])
    tag = "real" if int(crow["label"]) == 0 else "fake"
    mtype = crow["manipulation_type"] if "manipulation_type" in df.columns else ""
    b1.success(f"**{sel_id}** — {mtype} ({tag})")
    if b2.button("Choose clip", key="open_picker_btn"):
        st.session_state["show_picker"] = True

    @st.dialog("Choose a clip", width="large")
    def _picker():
        query = st.text_input("Search clip id / identity", key="pick_search")
        has_type = "manipulation_type" in df.columns
        has_method = "method" in df.columns
        f1, f2, f3 = st.columns(3)
        manip = (f1.multiselect("Type", MANIP_TYPES, key="pick_types") if has_type else [])
        methods = (f2.multiselect("Method", sorted(df["method"].dropna().unique().tolist()),
                                  key="pick_methods") if has_method else [])
        label_filter = f3.radio("Label", ["all", "real", "fake"], horizontal=True, key="pick_label")

        filtered = group_by_identity(search_manifest(
            filter_manifest(df, manip, methods, label_filter), query))
        st.caption(f"{len(filtered)} clips — variants of the same identity are grouped. "
                   "Click a row to use it.")
        view_cols = [c for c in ["clip_id", "source", "manipulation_type", "method", "label"]
                     if c in filtered.columns]
        event = st.dataframe(filtered[view_cols], height=380, hide_index=True,
                             on_select="rerun", selection_mode="single-row", key="pick_table")
        rows = event.selection["rows"]
        if rows:
            st.session_state["sel_clip_id"] = filtered.iloc[rows[0]]["clip_id"]
            st.session_state["show_picker"] = False
            st.rerun()

    if st.session_state.get("show_picker"):
        _picker()

    return df[df["clip_id"] == sel_id].iloc[0]
