"""Reusable Streamlit pieces for the stream pages: weights, and disabled model boxes.

The dashboard does not train. It never did, and it no longer offers to: the
Train tab that used to emit a background-trainer command is gone, because a
command builder is not a training feature and having one here suggested the
section was where training lived. Runs happen through `ddf run` and are tracked
in the local MLflow store. What comes back is a checkpoint, and this module is
where a page picks one.

Everything that renders a *model running* now lives on the per-stream subpages,
which show the forward pass step by step rather than reporting one number.
"""

from pathlib import Path

from deepfake_detection.dashboard.lib import checkpoints
from deepfake_detection.dashboard.paths import PROJECT_ROOT

# The Architecture control's labels, so the picker can name the setting to
# change rather than describe the tensor shapes that did not fit.
_TEMPORAL_LABEL = {
    ("lstm", True): "BiLSTM",
    ("gru", True): "GRU",
    ("lstm", False): "LSTM (unidirectional)",
    ("gru", False): "GRU (unidirectional)",
}


def _label(path: Path) -> str:
    """A checkpoint's name prefixed by the run that wrote it, when it sits in one."""
    try:
        relative = path.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        return path.name
    parts = relative.parts
    if parts[0] == "runs" and len(parts) > 2:
        return f"{parts[1]} / {path.name}"
    return path.name


def render_checkpoint_picker(
    st, stream_name: str, ns: str, store: dict | None = None
) -> Path | None:
    """Choose trained weights for a stream. Returns the chosen file, or None.

    None means untrained random weights, which is the honest default until a
    training run comes back: nothing on disk is a normal state, not an error.

    `store` is a sticky.run_state dict. When given, the selection survives a page
    switch, which Streamlit would otherwise reset by discarding the widget state.
    A remembered file that has since left the disk drops back to untrained.
    """
    found = checkpoints.discover(stream_name)
    # Labelled by run directory, not by filename: `ddf run` writes the same
    # `fold0-visual.pt` into every run it is pointed at, so the bare names
    # collide and the picker would silently keep only the newest of each.
    labels = {checkpoints.UNTRAINED: None} | {_label(p): p for p in found}

    names = list(labels)
    remembered = (store or {}).get("ckpt_choice")
    index = names.index(remembered) if remembered in names else 0

    c1, c2 = st.columns([2, 3])
    choice = c1.selectbox(
        "Checkpoint",
        names,
        index=index,
        key=f"{ns}_ckpt",
        help=f"Files under `checkpoints/{stream_name}/` and inside `runs/`, "
        "newest first, labelled by the run that wrote them.",
    )
    reference = c2.text_input(
        "or an MLflow run",
        key=f"{ns}_mlflow",
        value=(store or {}).get("mlflow_ref", ""),
        placeholder="run id, or runs:/<run id>/visual-initial.pt",
        help="Pulled from the local MLflow store and cached under "
        "checkpoints/_mlflow/. Takes precedence over the selection on the left.",
    )
    if store is not None:
        store.update(ckpt_choice=choice, mlflow_ref=reference)

    if reference.strip():
        try:
            with st.spinner("Downloading artifact..."):
                path = checkpoints.from_mlflow(reference.strip())
            st.caption(f"Pulled `{path.name}` from `{reference.strip()}`.")
            return path
        except (OSError, RuntimeError, ValueError) as error:
            st.error(f"Could not pull that run's artifact: {error}")
            return None

    path = labels[choice]
    if path is None:
        if not found:
            st.caption(
                f"No checkpoints for `{stream_name}` under `checkpoints/` or `runs/`. "
                "The weights are random, so the probability below is a plumbing "
                "check and not a detection."
            )
        return None

    info = checkpoints.describe(path)
    if info["error"]:
        st.error(f"`{path.name}` could not be read: {info['error']}")
        return None
    detail = f"{info['tensors']} tensors"
    if info["metadata"]:
        meta = info["metadata"]
        detail += f"  ·  run `{meta.get('run_id', '?')}`  ·  seed {meta.get('seed', '?')}"
    if info["config"]:
        detail += f"  ·  saved config: `{info['config']}`"
    st.caption(f"`{path}`  ·  {detail}")

    architecture = info["architecture"]
    if architecture and architecture["hidden"]:
        wanted = _TEMPORAL_LABEL.get(
            (architecture["temporal"], architecture["bidirectional"]),
            architecture["temporal"],
        )
        width = architecture.get("common_dim")
        st.caption(
            f"Trained with **{wanted}** at hidden **{architecture['hidden']}**"
            + (f", embedding **{width}**" if width else "")
            + ". The Architecture controls have been set to match; change them "
            "there to try this checkpoint against a different temporal model."
        )
    return path


# The head a Design A branch carries, against the two a Design B stream carries.
# `ddf train visual` ends in a classifier straight to one logit; a stream ends in
# a projection to the shared width, with the head only for development. So these
# never transfer, and saying "part of the model is randomly initialised" without
# saying which part reads as a configuration error when it is a real difference
# between the two designs.
_BRANCH_HEAD = {"classifier.weight", "classifier.bias"}
_STREAM_HEAD = {
    "projection.0.weight",
    "projection.0.bias",
    "projection.1.weight",
    "projection.1.bias",
    "temp_head.weight",
    "temp_head.bias",
}


def report_load(st, report: dict):
    """Say what the checkpoint did to the model. Silence would be worse than noise."""
    if report["clean"]:
        st.success(
            f"Loaded {report['matched']} tensors. The checkpoint matches this "
            "architecture exactly."
        )
        return
    if (
        not report["mismatched"]
        and set(report["missing"]) <= _STREAM_HEAD
        and set(report["unexpected"]) <= _BRANCH_HEAD
    ):
        st.success(
            f"Loaded {report['matched']} tensors: the backbone and the temporal "
            "model are the trained ones."
        )
        st.info(
            "The head did not transfer, and cannot. This checkpoint comes from "
            "`ddf train visual`, which ends in a classifier straight to one logit, "
            "while a stream ends in a projection to the shared width with the head "
            "only for development. So the features below are trained and the "
            "probability at the end is not."
        )
        return
    st.warning(
        f"Loaded {report['matched']} tensors, but the checkpoint does not match this "
        "configuration, so part of the model is still randomly initialised. Set the "
        "controls above to the architecture it was trained with."
    )
    if report["mismatched"]:
        lines = "\n".join(
            f"{name}: checkpoint {have} vs model {want}"
            for name, have, want in report["mismatched"][:8]
        )
        st.code(lines, language="text")
    for label, keys in (
        ("missing from the file", report["missing"]),
        ("in the file but not the model", report["unexpected"]),
    ):
        if keys:
            st.caption(
                f"{len(keys)} tensors {label}: `{', '.join(keys[:4])}`"
                f"{' …' if len(keys) > 4 else ''}"
            )


def render_disabled_model_box(st, name: str, role: str, note: str, key: str):
    """An encoder that is not built yet, shown as the box it will become.

    Greyed rather than absent, because the shape of the stream is the point of
    the page even before the weights exist.
    """
    with st.container(border=True):
        head, cfg = st.columns([1, 2])
        with head:
            st.markdown(f"### {name}")
            st.caption(role)
        with cfg:
            c1, c2 = st.columns(2)
            c1.selectbox(
                "Weights", ["(not downloaded)"], key=f"{key}_weights", disabled=True
            )
            c2.slider(
                "Embedding dim", 128, 512, 256, step=64, key=f"{key}_dim", disabled=True
            )
            st.button(
                "Run encoder",
                key=f"{key}_run",
                disabled=True,
                help=note,
                width="stretch",
            )
        st.caption(note)
