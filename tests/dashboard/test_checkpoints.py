"""Checkpoint discovery and loading, against real torch.save output on tmp_path."""
import pytest
import torch

from dashboard.lib import checkpoints
from models.streams.common.config import StreamConfig
from models.streams.common.visual_stream import build_visual_stream


def _model(**kw):
    base = dict(pretrained=False, grad_checkpointing=False, frame_chunk_size=0,
                temporal_hidden=64, common_dim=128)
    base.update(kw)
    return build_visual_stream(StreamConfig(**base))


def test_discover_returns_nothing_when_no_directory_exists(tmp_path):
    """No checkpoints is the normal state until a training run comes back."""
    assert checkpoints.discover("xception", root=tmp_path) == []


def test_discover_lists_newest_first(tmp_path):
    directory = tmp_path / "xception"
    directory.mkdir()
    for name, mtime in [("old.pt", 1_000), ("new.pt", 9_000), ("middle.ckpt", 5_000)]:
        path = directory / name
        path.write_bytes(b"x")
        import os
        os.utime(path, (mtime, mtime))
    found = checkpoints.discover("xception", root=tmp_path)
    assert [p.name for p in found] == ["new.pt", "middle.ckpt", "old.pt"]


def test_discover_ignores_files_that_are_not_checkpoints(tmp_path):
    directory = tmp_path / "xception"
    directory.mkdir()
    (directory / "weights.pt").write_bytes(b"x")
    (directory / "notes.md").write_text("not a checkpoint")
    assert [p.name for p in checkpoints.discover("xception", root=tmp_path)] == ["weights.pt"]


def test_describe_reads_a_bare_state_dict(tmp_path):
    model = _model()
    path = tmp_path / "bare.pt"
    torch.save(model.state_dict(), path)
    info = checkpoints.describe(path)
    assert info["error"] is None
    assert info["tensors"] == len(model.state_dict())
    assert info["config"] is None


def test_describe_reads_a_wrapped_checkpoint_and_its_config(tmp_path):
    """The trainer must save its config as a plain dict: weights_only refuses objects."""
    model = _model()
    path = tmp_path / "wrapped.pt"
    torch.save({"state_dict": model.state_dict(), "config": {"temporal_hidden": 64}}, path)
    info = checkpoints.describe(path)
    assert info["error"] is None
    assert info["config"] == {"temporal_hidden": 64}


def test_describe_reports_an_unreadable_file_instead_of_raising(tmp_path):
    """The page has to render whatever is in the directory, including junk."""
    path = tmp_path / "corrupt.pt"
    path.write_bytes(b"not a torch file at all")
    info = checkpoints.describe(path)
    assert info["error"]
    assert info["tensors"] == 0


def test_load_into_a_matching_model_is_clean(tmp_path):
    path = tmp_path / "match.pt"
    torch.save(_model().state_dict(), path)
    report = checkpoints.load_into(_model(), path)
    assert report["clean"]
    assert report["missing"] == [] and report["unexpected"] == []
    assert report["matched"] > 0


def test_load_into_reports_surplus_and_absent_keys(tmp_path):
    """A different temporal model, so its weights have nowhere to go."""
    path = tmp_path / "other.pt"
    torch.save(_model(temporal_type="lstm").state_dict(), path)
    report = checkpoints.load_into(_model(temporal_type="mean"), path)
    assert not report["clean"]
    assert any("temporal" in key for key in report["unexpected"])


def test_load_into_reports_a_shape_clash_instead_of_raising(tmp_path):
    """strict=False tolerates absent and surplus keys but still raises on a size
    mismatch, which is exactly what a changed hidden size produces."""
    path = tmp_path / "wide.pt"
    torch.save(_model(temporal_hidden=64).state_dict(), path)
    report = checkpoints.load_into(_model(temporal_hidden=32), path)
    assert not report["clean"]
    clashed = {name for name, _from, _to in report["mismatched"]}
    assert any("temporal" in name for name in clashed)
    assert report["matched"] > 0, "the tensors that did fit are still loaded"


def test_load_into_actually_moves_the_weights(tmp_path):
    trained, fresh = _model(), _model()
    path = tmp_path / "w.pt"
    torch.save(trained.state_dict(), path)
    before = fresh.temp_head.weight.clone()
    checkpoints.load_into(fresh, path)
    assert not torch.equal(before, fresh.temp_head.weight)
    assert torch.equal(trained.temp_head.weight, fresh.temp_head.weight)


def test_from_wandb_explains_itself_when_wandb_is_absent(monkeypatch):
    """Deferred import, so the dashboard runs without wandb; the message must say so."""
    import builtins
    real_import = builtins.__import__

    def _no_wandb(name, *args, **kwargs):
        if name == "wandb":
            raise ImportError("no wandb")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_wandb)
    with pytest.raises(RuntimeError, match="wandb is not installed"):
        checkpoints.from_wandb("entity/project/model:v0")
