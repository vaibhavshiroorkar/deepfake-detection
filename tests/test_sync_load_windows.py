"""load_windows leaves out windows whose video ended early, instead of failing on the stack."""

import csv
import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_sync_path.py"


def _module():
    spec = importlib.util.spec_from_file_location("train_sync_path", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_short_windows_are_left_out(tmp_path, capsys):
    from deepfake_detection.avsync.features import cache_path

    module = _module()
    rows = []
    for clip, frames in (("a", 50), ("b", 31), ("c", 50)):
        window = f"{clip}-w0"
        path = cache_path(tmp_path, window)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = np.zeros((frames, 768), np.float16)
        np.savez(path, lips=data, audio=data, fused=data, face=np.float32(1.0))
        rows.append(
            {
                "clip_id": clip,
                "window_id": window,
                "start": 0,
                "file": "",
                "face": 1.0,
                "status": "ok",
            }
        )
    with (tmp_path / "index.csv").open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    owners, lips, audio = module.load_windows(tmp_path, {"a": {}, "b": {}, "c": {}})

    assert owners == ["a", "c"]
    assert lips.shape == (2, 50, 768) and audio.shape == (2, 50, 768)
    assert "left out 1 windows" in capsys.readouterr().out
