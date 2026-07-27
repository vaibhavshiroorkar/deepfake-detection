"""Shared Dataset -> Split -> Target -> Clip selection for the preprocessing page.

Selection is two-level: pick a DATASET, then a SPLIT within it. Neither is
hardcoded — datasets.discover() scans data/ on load, so dropping a dataset
folder in and hitting the ↻ button is all it takes to work with it (see
dashboard/lib/datasets.py for the detection rules).

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

from dashboard.lib import datasets

DATA_DIR = _REPO_ROOT / "data"

MANIP_TYPES = ["RealVideo-RealAudio", "RealVideo-FakeAudio",
               "FakeVideo-RealAudio", "FakeVideo-FakeAudio"]


def registry(refresh: bool = False) -> dict[str, datasets.Dataset]:
    """Discovered datasets, scanned once per session (or on ↻)."""
    import streamlit as st

    if refresh or "ds_registry" not in st.session_state:
        st.session_state["ds_registry"] = datasets.discover(DATA_DIR)
        st.session_state["ds_manifests"] = {}
    return st.session_state["ds_registry"]


def load_manifest(dataset: str, split: str) -> pd.DataFrame:
    """Manifest frame for a discovered dataset/split, memoized per session.

    Worth memoizing: a raw drop's manifest is derived from meta_data.csv and
    stats ~21k files, which is far too slow to redo on every Streamlit rerun.
    """
    import streamlit as st

    ds = registry().get(dataset)
    if ds is None:
        raise KeyError(f"Unknown dataset {dataset!r} — not found under {DATA_DIR}")
    cache = st.session_state.setdefault("ds_manifests", {})
    key = (dataset, split)
    if key not in cache:
        with st.spinner(f"Reading {dataset} / {split}…"):
            cache[key] = datasets.load_split(ds, split, DATA_DIR)
    return cache[key]


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


def _keep_valid(key: str, options: list[str]):
    """Drop a stored widget choice that a rescan removed, so the widget resets."""
    import streamlit as st

    if key in st.session_state and st.session_state[key] not in options:
        del st.session_state[key]


def render_selection():
    """Render Dataset/Split + a modal Clip picker; return the selected clip row."""
    import streamlit as st

    st.header("Selection")
    c1, c2, c3 = st.columns([6, 6, 1], vertical_alignment="bottom")
    with c3:
        if st.button("↻", key="sel_refresh", help="Rescan data/ for datasets"):
            registry(refresh=True)
            st.rerun()

    found = registry()
    if not found:
        st.info(f"No datasets found in `{DATA_DIR}`. Drop a dataset folder there "
                "(a FakeAVCeleb-style `meta_data.csv`, or manifest CSVs with "
                "clip_id/video_path/label columns) and hit ↻.")
        return None

    names = list(found)
    _keep_valid("sel_dataset", names)
    with c1:
        dataset = st.selectbox("Dataset", names, key="sel_dataset")

    splits = found[dataset].splits
    _keep_valid("sel_split", splits)
    with c2:
        split = st.selectbox("Split", splits, key="sel_split")

    df = load_manifest(dataset, split)
    st.caption(f"`{found[dataset].root}` — {len(df)} clips in *{split}*.")
    if df.empty:
        st.warning("This split has no clips on disk.")
        return None

    # Reset the chosen clip whenever the dataset/split changes.
    ctx = f"{dataset}/{split}"
    if st.session_state.get("sel_context") != ctx:
        st.session_state["sel_context"] = ctx
        st.session_state.pop("sel_clip_id", None)

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
    # Call the dialog directly on the click. A persistent flag would re-open it
    # on every rerun (e.g. switching to the Config tab) after a dismiss.
    open_clicked = b2.button("Choose clip", key="open_picker_btn")

    @st.dialog("Choose a clip", width="large")
    def _picker():
        left, right = st.columns([3, 2])
        with left:
            query = st.text_input("Search clip id / identity", key="pick_search")
            has_type = "manipulation_type" in df.columns
            has_method = "method" in df.columns
            f1, f2, f3 = st.columns(3)
            manip = (f1.multiselect("Type", MANIP_TYPES, key="pick_types") if has_type else [])
            methods = (f2.multiselect("Method", sorted(df["method"].dropna().unique().tolist()),
                                      key="pick_methods") if has_method else [])
            label_filter = f3.radio("Label", ["all", "real", "fake"], horizontal=True,
                                    key="pick_label")
            filtered = group_by_identity(search_manifest(
                filter_manifest(df, manip, methods, label_filter), query))
            st.caption(f"{len(filtered)} clips — variants of one identity are grouped. "
                       "Select a row to preview it on the right.")
            view_cols = [c for c in ["clip_id", "source", "manipulation_type", "method", "label"]
                         if c in filtered.columns]
            event = st.dataframe(filtered[view_cols], height=340, hide_index=True,
                                 on_select="rerun", selection_mode="single-row", key="pick_table")
            rows = event.selection["rows"]
        candidate = filtered.iloc[rows[0]] if rows else None
        with right:
            if candidate is None:
                st.info("Select a clip on the left to preview it.")
            else:
                ctag = "real" if int(candidate["label"]) == 0 else "fake"
                cmt = candidate["manipulation_type"] if "manipulation_type" in candidate else ""
                st.markdown(f"**{candidate['clip_id']}**")
                st.caption(f"{cmt} ({ctag})")
                _preview_frames(st, DATA_DIR / candidate["video_path"])
                if st.button("Use this clip", type="primary", key="pick_confirm"):
                    st.session_state["sel_clip_id"] = candidate["clip_id"]
                    st.rerun()

    if open_clicked:
        _picker()

    return df[df["clip_id"] == sel_id].iloc[0]


def _preview_frames(st, video_path, n: int = 4):
    """Show a few raw frames of a clip inside the picker (no MTCNN)."""
    import cv2
    from dashboard.lib import media

    if not Path(video_path).exists():
        st.caption("Video not found on disk.")
        return
    duration, _ = media.frame_meta(str(video_path))
    ts = media.sample_timestamps(duration, n, 0.35)
    frames = [cv2.resize(f, (150, 150)) for f in media.decode_frames(str(video_path), ts)]
    cols = st.columns(2)
    for k, im in enumerate(frames):
        cols[k % 2].image(im, width="stretch")
