"""Find and load trained weights, so the Streams pages can show a real model.

Training happens elsewhere (PROJECT_OVERVIEW.md section 7: Kaggle or a GPU box,
tracked in W&B). This module is the other end of that trip: it lists whatever
checkpoints have made it back to `checkpoints/<stream_name>/`, or pulls one from
a W&B artifact, and loads it into a stream the dashboard just built.

Loading is deliberately non-strict *and* loud. A checkpoint trained with a
BiLSTM at 256 hidden will not fit a model configured for a GRU at 128, and the
useful behaviour is to say which tensors did not fit rather than either crashing
or quietly leaving half the network at its random initialisation.

`describe` reads with weights_only=True, which refuses to unpickle arbitrary
objects. A trainer that wants its config to survive the trip must therefore save
it as a plain dict, not as a StreamConfig instance.
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

CHECKPOINT_DIR = _REPO_ROOT / "checkpoints"
SUFFIXES = (".pt", ".pth", ".ckpt")

# The option that is always present, because no checkpoint is a normal state:
# nothing is trained yet, and a stream still builds and runs.
UNTRAINED = "(untrained, random weights)"

# Keys a checkpoint might nest its weights under. A file that is just a
# state_dict is also accepted, which is what `torch.save(model.state_dict())`
# produces and what anyone hand-saving from a notebook will write.
STATE_KEYS = ("state_dict", "model_state_dict", "model", "weights")


def discover(stream_name: str, root: Path | None = None) -> list[Path]:
    """Checkpoints for one stream, newest first. Missing directory means none."""
    directory = (root or CHECKPOINT_DIR) / stream_name
    if not directory.is_dir():
        return []
    found = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in SUFFIXES]
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


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

    out = {"path": Path(path), "tensors": 0, "config": None, "error": None}
    try:
        blob = torch.load(path, map_location="cpu", weights_only=True)
        state = _state_dict(blob)
        out["tensors"] = len(state)
        if isinstance(blob, dict) and isinstance(blob.get("config"), dict):
            out["config"] = blob["config"]
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

    mismatched = [(k, tuple(v.shape), tuple(current[k].shape)) for k, v in state.items()
                  if k in current and tuple(v.shape) != tuple(current[k].shape)]
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


def from_wandb(reference: str, cache_dir: Path | None = None) -> Path:
    """Download a W&B artifact and return the checkpoint file inside it.

    `reference` is what the run page shows, `entity/project/name:version`. The
    import is deferred so the dashboard starts in an environment without wandb
    installed, and says so here rather than failing at import time.
    """
    try:
        import wandb
    except ImportError as e:
        raise RuntimeError(
            "wandb is not installed in this environment, so a W&B artifact cannot be "
            "pulled. Install it with `uv sync`, or point at a local file under "
            "checkpoints/ instead.") from e

    target = cache_dir or (CHECKPOINT_DIR / "_wandb")
    target.mkdir(parents=True, exist_ok=True)
    artifact = wandb.Api().artifact(reference)
    root = Path(artifact.download(root=str(target / artifact.name.replace(":", "_"))))

    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUFFIXES]
    if not files:
        raise RuntimeError(f"artifact {reference} holds no {'/'.join(SUFFIXES)} file")
    return max(files, key=lambda p: p.stat().st_size)
