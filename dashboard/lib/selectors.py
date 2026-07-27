"""Clip selection: a dataset clip, or a video you upload.

Two sources, chosen by a segmented control and returning the same shape of row
either way, so callers do not care which was used:

  - **Dataset**: pick a DATASET, then a SPLIT within it, then a clip through the
    picker dialog. Neither list is hardcoded: datasets.discover() scans data/ on
    load, so dropping a dataset folder in and hitting ↻ is all it takes (see
    dashboard/lib/datasets.py for the detection rules).
  - **Upload**: hand it any mp4. It is written to a temp directory so OpenCV and
    PyAV have a real path to open, and never into data/.

Callers must resolve a row's clip through clip_path(), not by joining DATA_DIR
themselves: an uploaded clip's video_path is absolute and lives outside data/.

Pure filtering (filter_manifest, search_manifest, group_by_identity) is separated
from the Streamlit widgets (render_selection) so the filter logic stays
unit-testable without a running app. All shared st.session_state keys are owned
here.
"""
import sys
import tempfile
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dashboard.lib import datasets

DATA_DIR = _REPO_ROOT / "data"

MANIP_TYPES = ["RealVideo-RealAudio", "RealVideo-FakeAudio",
               "FakeVideo-RealAudio", "FakeVideo-FakeAudio"]

SOURCE_DATASET = "Dataset"
SOURCE_UPLOAD = "Upload a video"

# Uploads land outside the repo on purpose: data/ belongs to the dataset, and
# the dashboard is read-only with respect to it.
UPLOAD_DIR = Path(tempfile.gettempdir()) / "deepfake_dashboard_uploads"

# An uploaded clip has no ground truth. -1 keeps the column integer-typed while
# staying an obvious non-label; label_text() renders it as "unknown".
LABEL_UNKNOWN = -1


def clip_path(row) -> Path:
    """Absolute path to a selected clip, dataset row or upload alike.

    A dataset row's video_path is relative to data/; an upload's is already
    absolute. Joining DATA_DIR unconditionally happens to work on Windows, but
    only by accident of pathlib's absolute-path rule, so it is done explicitly.
    """
    path = Path(row["video_path"])
    return path if path.is_absolute() else DATA_DIR / path


def label_text(row) -> str:
    """'real', 'fake', or 'unknown' for an upload with no ground truth."""
    label = int(row["label"])
    if label == LABEL_UNKNOWN:
        return "unknown"
    return "real" if label == 0 else "fake"


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
        raise KeyError(f"Unknown dataset {dataset!r}, not found under {DATA_DIR}")
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


def upload_row(name: str, data: bytes, file_id: str) -> pd.Series:
    """A selection row for an uploaded video, written to UPLOAD_DIR.

    Takes the bytes rather than a Streamlit UploadedFile so it is unit-testable.
    file_id keys the temp copy, so a rerun does not rewrite it and two uploads
    sharing a filename do not collide.
    """
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / f"{file_id}_{Path(name).name}"
    if not dest.exists():
        dest.write_bytes(data)
    return pd.Series({
        "clip_id": Path(name).stem,
        "video_path": str(dest),
        "label": LABEL_UNKNOWN,
        "manipulation_type": "uploaded",
        "method": "unknown",
        "source": "upload",
    })


def _keep_valid(key: str, options: list[str]):
    """Drop a stored widget choice that a rescan removed, so the widget resets."""
    import streamlit as st

    if key in st.session_state and st.session_state[key] not in options:
        del st.session_state[key]


def render_selection():
    """Render the clip source controls; return the selected clip row, or None."""
    import streamlit as st

    st.header("Selection")
    source = st.segmented_control(
        "Clip source", [SOURCE_DATASET, SOURCE_UPLOAD], default=SOURCE_DATASET,
        key="sel_source", help="Browse a discovered dataset, or bring your own video.")
    if source == SOURCE_UPLOAD:
        return _render_upload(st)
    return _render_dataset(st)


# --------------------------------------------------------------------- upload

def _render_upload(st):
    uploaded = st.file_uploader(
        "Video file", type=["mp4", "mov", "mkv", "webm", "avi"], key="sel_upload",
        help="Decoded and previewed exactly like a dataset clip. Never copied into data/.")
    if uploaded is None:
        st.caption("Upload a video to run it through the same preprocessing steps. It carries no "
                   "ground-truth label, so its label reads *unknown*.")
        return None

    row = upload_row(uploaded.name, uploaded.getvalue(), str(uploaded.file_id))
    _selected_card(st, row, action_label=None)
    return row


# -------------------------------------------------------------------- dataset

def _render_dataset(st):
    c1, c2, c3 = st.columns([6, 6, 1], vertical_alignment="bottom")
    with c3:
        if st.button("↻", key="sel_refresh", help="Rescan data/ for datasets"):
            registry(refresh=True)
            st.rerun()

    found = registry()
    if not found:
        st.info(f"No datasets found in `{DATA_DIR}`. Drop a dataset folder there "
                "(a FakeAVCeleb-style `meta_data.csv`, or manifest CSVs with "
                "clip_id/video_path/label columns) and hit ↻. Or switch to "
                f"**{SOURCE_UPLOAD}** to preview a single file.")
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
    st.caption(f"`{found[dataset].root}` · {len(df)} clips in *{split}*.")
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
    if _selected_card(st, crow, action_label="Choose clip"):
        st.session_state["sel_picker_open"] = True

    # The dialog's open state lives in session_state rather than being driven by
    # the button click directly. A widget inside a dialog reruns the script, and
    # a dialog that is not re-invoked on that rerun disappears. on_dismiss clears
    # the flag, so an X or ESC does not leave it set to pop the dialog open again
    # at the next unrelated interaction.
    if st.session_state.get("sel_picker_open"):
        _picker(st, df)

    return df[df["clip_id"] == sel_id].iloc[0]


def _close_picker():
    import streamlit as st

    st.session_state["sel_picker_open"] = False


def _selected_card(st, row, action_label: str | None):
    """The current selection, shown plainly. Returns True if the action was clicked.

    A bordered container rather than st.success: the selection is a statement of
    fact, not an achievement, and a full-width green banner reads as one on every
    single rerun.
    """
    with st.container(border=True):
        info, action = ((st.container(), None) if action_label is None
                        else st.columns([5, 1], vertical_alignment="center"))
        info.markdown(f"**{row['clip_id']}**")
        mtype = row["manipulation_type"] if "manipulation_type" in row else "unknown"
        method = row["method"] if "method" in row else "unknown"
        info.caption(f"{mtype}  ·  {method}  ·  label **{label_text(row)}**")
        if action is None:
            return False
        return action.button(action_label, key="open_picker_btn", width="stretch")


def _picker(st, df):
    """The clip picker dialog. Stays open until dismissed, so several can be tried."""

    @st.dialog("Choose a clip", width="large", on_dismiss=_close_picker)
    def _body():
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
            st.caption(f"{len(filtered)} clips. Variants of one identity are grouped. "
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
                st.markdown(f"**{candidate['clip_id']}**")
                cmt = candidate["manipulation_type"] if "manipulation_type" in candidate else ""
                st.caption(f"{cmt} ({label_text(candidate)})")
                _preview_frames(st, clip_path(candidate))
                # Applying does NOT close the dialog. Comparing a few clips in one
                # sitting is how this actually gets used, and reopening the picker
                # for each one threw away the filters and the scroll position.
                if st.button("Use this clip", type="primary", key="pick_confirm",
                             width="stretch"):
                    st.session_state["sel_clip_id"] = candidate["clip_id"]
                if st.session_state.get("sel_clip_id") == candidate["clip_id"]:
                    st.caption("Selected. Close this dialog to preview it on the page.")

    _body()


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
