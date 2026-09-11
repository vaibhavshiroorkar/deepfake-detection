"""Find and load trained weights, so the Streams pages can show a real model.

Training happens through `ddf run`, tracked in the local MLflow store. This
module is the other end of that trip: it lists whatever checkpoints have made it
back to `checkpoints/<stream_name>/`, or pulls one from an MLflow run's
artifacts, and loads it into a stream the dashboard just built.

Loading is deliberately non-strict *and* loud. A checkpoint trained with a
BiLSTM at 256 hidden will not fit a model configured for a GRU at 128, and the
useful behaviour is to say which tensors did not fit rather than either crashing
or quietly leaving half the network at its random initialisation.

`describe` reads with weights_only=True, which refuses to unpickle arbitrary
objects. A trainer that wants its config to survive the trip must therefore save
it as a plain dict, not as a StreamConfig instance.
"""

from pathlib import Path

from deepfake_detection.dashboard.paths import CHECKPOINT_DIR, RUNS_DIR

SUFFIXES = (".pt", ".pth", ".ckpt")

# The option that is always present, because no checkpoint is a normal state:
# nothing is trained yet, and a stream still builds and runs.
UNTRAINED = "(untrained, random weights)"

# Keys a checkpoint might nest its weights under. A file that is just a
# state_dict is also accepted, which is what `torch.save(model.state_dict())`
# produces and what anyone hand-saving from a notebook will write.
# "model_state" is what `ddf train visual` writes, so it comes
# first: every checkpoint this project produces is nested under it. The rest are
# the conventions other trainers use, kept so a checkpoint from elsewhere loads.
STATE_KEYS = ("model_state", "state_dict", "model_state_dict", "model", "weights")


# The token that names a stream in a checkpoint filename.
STREAM_TOKENS = {
    "efficientnet": "efficientnet",
    "xception": "xception",
    "dinov3": "dinov3",
    "lipsync": "lipsync",
    "emotion": "emotion",
}

# `ddf train visual` is welded to EfficientNet-B0 and names its output "visual"
# with no backbone in it, so without this every Design A visual checkpoint stays
# invisible. It cannot be a plain alias: `ddf train visual-stream` writes
# `visual-dinov3.pt` and `visual-xception.pt`, which also contain "visual", so a
# generic token only applies to a file that names no other stream.
GENERIC_TOKENS = {"efficientnet": ("visual",)}


def _belongs(path: Path, stream_name: str) -> bool:
    """Does this checkpoint file belong to this stream?

    A specific token wins outright. A generic one only counts when the filename
    names no other stream, so `visual-dinov3.pt` goes to dinov3 rather than to
    both dinov3 and efficientnet.
    """
    name = path.name.lower()
    own = STREAM_TOKENS.get(stream_name, stream_name)
    if own in name:
        return True
    others = {token for key, token in STREAM_TOKENS.items() if key != stream_name}
    if any(token in name for token in others):
        return False
    return any(token in name for token in GENERIC_TOKENS.get(stream_name, ()))


def discover(
    stream_name: str, root: Path | None = None, runs_root: Path | None = None
) -> list[Path]:
    """Checkpoints for one stream, newest first. Nothing found means none.

    Two places, because training does not write where the dashboard used to
    look. `ddf run` leaves its weights inside the run directory it was given,
    under `runs/<run>/checkpoints/`, and nothing ever copies them to the
    top-level `checkpoints/<stream>/` this function started with. That directory
    has never existed in this repository, so the picker offered only untrained
    weights while twenty trained checkpoints sat on disk.
    """
    found: list[Path] = []
    directory = (root or CHECKPOINT_DIR) / stream_name
    if directory.is_dir():
        found += [
            p
            for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in SUFFIXES
        ]

    runs = runs_root or RUNS_DIR
    if runs.is_dir():
        # Two shallow globs, not rglob. A run directory holds its preprocessed
        # cache beside its weights, so walking the tree visits about 62,000
        # entries and takes ten seconds, once per Streamlit rerun, which reads
        # on screen as a page that stops rendering at the checkpoint picker.
        # These two patterns find the same 32 files in three milliseconds.
        candidates = list(runs.glob("*/checkpoints/*")) + list(runs.glob("*/*"))
        found += [
            p
            for p in candidates
            if p.is_file() and p.suffix.lower() in SUFFIXES and _belongs(p, stream_name)
        ]

    # Two runs can leave identically named files, so the path is the identity
    # rather than the name, and the picker labels them by path further down.
    unique = {p.resolve(): p for p in found}
    return sorted(unique.values(), key=lambda p: p.stat().st_mtime, reverse=True)


def architecture(state: dict) -> dict:
    """What temporal model a checkpoint's tensors imply, read from their shapes.

    Needed because the dashboard defaults to a BiLSTM and every trained visual
    checkpoint here used a unidirectional GRU. Loading is non-strict, so the
    mismatch is reported rather than raised, but "these four tensors are the
    wrong shape" does not tell you which control to move. This does.
    """
    weight = state.get("temporal.weight_hh_l0")
    if weight is None:
        return {"temporal": "mean", "hidden": None, "bidirectional": False}
    hidden = int(weight.shape[1])
    gates = int(weight.shape[0]) // max(hidden, 1)
    return {
        # A GRU holds three gate matrices per layer, an LSTM four.
        "temporal": {3: "gru", 4: "lstm"}.get(gates, f"{gates}-gate"),
        "hidden": hidden,
        "bidirectional": "temporal.weight_hh_l0_reverse" in state,
    }


def _state_dict(obj) -> dict:
    """The tensor mapping inside a loaded checkpoint, whatever it was wrapped in."""
    if not isinstance(obj, dict):
        raise ValueError(f"checkpoint holds a {type(obj).__name__}, not a state dict")
    for key in STATE_KEYS:
        inner = obj.get(key)
        if isinstance(inner, dict) and inner:
            return inner
    return obj


def describe(path: Path) -> dict:
    """What a checkpoint file claims to be, without building a model for it.

    Never raises: an unreadable checkpoint is a thing the page has to render, so
    the failure comes back in `error` rather than taking the page down.
    """
    import torch

    out = {
        "path": Path(path),
        "tensors": 0,
        "config": None,
        "architecture": None,
        "metadata": None,
        "error": None,
    }
    try:
        blob = torch.load(path, map_location="cpu", weights_only=True)
        state = _state_dict(blob)
        out["tensors"] = len(state)
        out["architecture"] = architecture(state)
        if isinstance(blob, dict) and isinstance(blob.get("config"), dict):
            out["config"] = blob["config"]
        if isinstance(blob, dict) and isinstance(blob.get("metadata"), dict):
            out["metadata"] = blob["metadata"]
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def load_into(model, path: Path) -> dict:
    """Load a checkpoint into `model` and report exactly what did and did not fit.

    Three ways a tensor can fail to land, and all three are reported rather than
    raised: `missing` is in the model but not the file, `unexpected` is the other
    way round, and `mismatched` is present in both at different shapes, which is
    what a changed hidden size or embedding dim looks like.

    Same-name-different-shape has to be filtered out before the load, because
    strict=False tolerates absent and surplus keys but still raises on a size
    mismatch, and a raise here would just be a stack trace where the page needs a
    sentence about the config not matching.
    """
    import torch

    blob = torch.load(path, map_location="cpu", weights_only=True)
    state = _state_dict(blob)
    current = model.state_dict()

    mismatched = [
        (k, tuple(v.shape), tuple(current[k].shape))
        for k, v in state.items()
        if k in current and tuple(v.shape) != tuple(current[k].shape)
    ]
    loadable = {k: v for k, v in state.items() if k not in {m[0] for m in mismatched}}

    result = model.load_state_dict(loadable, strict=False)
    missing = [k for k in result.missing_keys if k not in {m[0] for m in mismatched}]
    unexpected = list(result.unexpected_keys)
    return {
        "missing": missing,
        "unexpected": unexpected,
        "mismatched": mismatched,
        "matched": len(loadable) - len(unexpected),
        "clean": not missing and not unexpected and not mismatched,
    }


def from_mlflow(reference: str, cache_dir: Path | None = None) -> Path:
    """Download an MLflow run's checkpoint artifact and return the file.

    `reference` is a run id, or a `runs:/<run_id>/<artifact path>` URI naming one
    file inside it. A bare run id downloads the run's whole artifact tree and
    picks the largest weight file in it, which is what a training run leaves
    behind.

    The import is deferred so the dashboard starts in an environment without
    mlflow installed, and says so here rather than at import time.
    """
    from deepfake_detection.dashboard.lib import mlflow_runs

    target = cache_dir or (CHECKPOINT_DIR / "_mlflow")
    target.mkdir(parents=True, exist_ok=True)
    root = Path(mlflow_runs.download_artifacts(reference, target))

    if root.is_file():
        return root
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUFFIXES
    ]
    if not files:
        raise RuntimeError(
            f"MLflow reference {reference} holds no {'/'.join(SUFFIXES)} file"
        )
    return max(files, key=lambda path: path.stat().st_size)
