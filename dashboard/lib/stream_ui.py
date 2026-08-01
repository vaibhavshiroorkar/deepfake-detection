"""Reusable Streamlit pieces for the stream pages: weights, and disabled model boxes.

The dashboard does not train (PROJECT_OVERVIEW.md section 7). It never did, and
it no longer offers to: the Train tab that used to emit a background-trainer
command is gone, because a command builder is not a training feature and having
one here suggested the section was where training lived. Runs happen on Kaggle
or a GPU box and are tracked in W&B. What comes back is a checkpoint, and this
module is where a page picks one.

Everything that renders a *model running* now lives on the per-stream subpages,
which show the forward pass step by step rather than reporting one number.
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dashboard.lib import checkpoints


def render_checkpoint_picker(st, stream_name: str, ns: str, store: dict | None = None
                             ) -> Path | None:
    """Choose trained weights for a stream. Returns the chosen file, or None.

    None means untrained random weights, which is the honest default until a
    training run comes back: nothing on disk is a normal state, not an error.

    `store` is a sticky.run_state dict. When given, the selection survives a page
    switch, which Streamlit would otherwise reset by discarding the widget state.
    A remembered file that has since left the disk drops back to untrained.
    """
    found = checkpoints.discover(stream_name)
    labels = {checkpoints.UNTRAINED: None} | {p.name: p for p in found}

    names = list(labels)
    remembered = (store or {}).get("ckpt_choice")
    index = names.index(remembered) if remembered in names else 0

    c1, c2 = st.columns([2, 3])
    choice = c1.selectbox("Checkpoint", names, index=index, key=f"{ns}_ckpt",
                          help=f"Files under `checkpoints/{stream_name}/`, newest first.")
    reference = c2.text_input(
        "or a W&B artifact", key=f"{ns}_wandb", value=(store or {}).get("wandb_ref", ""),
        placeholder="entity/project/name:v0",
        help="Pulled with the W&B API and cached under checkpoints/_wandb/. "
             "Takes precedence over the selection on the left.")
    if store is not None:
        store.update(ckpt_choice=choice, wandb_ref=reference)

    if reference.strip():
        try:
            with st.spinner("Downloading artifact…"):
                path = checkpoints.from_wandb(reference.strip())
            st.caption(f"Pulled `{path.name}` from `{reference.strip()}`.")
            return path
        except Exception as e:
            st.error(f"Could not pull that artifact: {e}")
            return None

    path = labels[choice]
    if path is None:
        if not found:
            st.caption(f"No checkpoints under `checkpoints/{stream_name}/`. The weights are "
                       "random, so the probability below is a plumbing check and not a "
                       "detection.")
        return None

    info = checkpoints.describe(path)
    if info["error"]:
        st.error(f"`{path.name}` could not be read: {info['error']}")
        return None
    detail = f"{info['tensors']} tensors"
    if info["config"]:
        detail += f"  ·  saved config: `{info['config']}`"
    st.caption(f"`{path}`  ·  {detail}")
    return path


def report_load(st, report: dict):
    """Say what the checkpoint did to the model. Silence would be worse than noise."""
    if report["clean"]:
        st.success(f"Loaded {report['matched']} tensors. The checkpoint matches this "
                   "architecture exactly.")
        return
    st.warning(
        f"Loaded {report['matched']} tensors, but the checkpoint does not match this "
        "configuration, so part of the model is still randomly initialised. Set the "
        "controls above to the architecture it was trained with.")
    if report["mismatched"]:
        lines = "\n".join(f"{name}: checkpoint {have} vs model {want}"
                          for name, have, want in report["mismatched"][:8])
        st.code(lines, language="text")
    for label, keys in (("missing from the file", report["missing"]),
                        ("in the file but not the model", report["unexpected"])):
        if keys:
            st.caption(f"{len(keys)} tensors {label}: `{', '.join(keys[:4])}`"
                       f"{' …' if len(keys) > 4 else ''}")


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
            c1.selectbox("Weights", ["(not downloaded)"], key=f"{key}_weights", disabled=True)
            c2.slider("Embedding dim", 128, 512, 256, step=64, key=f"{key}_dim", disabled=True)
            st.button("Run encoder", key=f"{key}_run", disabled=True, help=note,
                      width="stretch")
        st.caption(note)
