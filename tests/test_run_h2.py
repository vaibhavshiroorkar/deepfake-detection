"""End to end: run_h2.py on small synthetic prediction files."""

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_h2.py"


def _load():
    spec = importlib.util.spec_from_file_location("run_h2", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _fixture(tmp: Path, parts, drop_audio_clip=None):
    rng = np.random.default_rng(1)
    for part in parts:
        n = 160
        labels = rng.integers(0, 2, n)
        for seed in (17, 29, 43):
            noise = np.random.default_rng(seed).normal(0, 1, (n, 3))
            def prob(k, gain, labels=labels, noise=noise):
                return 1 / (1 + np.exp(-(labels * gain + noise[:, k])))
            base = [{"clip_id": f"{part}-{i}", "label": int(labels[i]), "source": f"id{i // 4}"} for i in range(n)]
            sync_dir = tmp / ("sync" if part in ("val", "test", "test-identity-strict") else "targets")
            _write(sync_dir / f"sync-{part}-seed{seed}-predictions.csv",
                   [{**b, "probability": f"{prob(2, 0.5)[i]:.6f}"} for i, b in enumerate(base)])
            _write(tmp / "branches" / f"visual-seed{seed}-{part}-predictions.csv",
                   [{**b, "probability": f"{prob(0, 0.7)[i]:.6f}"} for i, b in enumerate(base)])
            audio = [{**b, "probability": f"{prob(1, 2.0)[i]:.6f}"} for i, b in enumerate(base)]
            if drop_audio_clip is not None and seed == 29:
                audio = [a for a in audio if a["clip_id"] != f"{part}-{drop_audio_clip}"]
            _write(tmp / "branches" / f"audio-seed{seed}-{part}-predictions.csv", audio)


def test_run_h2_end_to_end(tmp_path):
    parts = ("val", "test", "test-identity-strict", "dfdc", "lavdf-test", "hifiavdf")
    _fixture(tmp_path, parts, drop_audio_clip=3)
    out = tmp_path / "out"
    code = _load().main(["--branch-dir", str(tmp_path / "branches"), "--sync-dir", str(tmp_path / "sync"),
                         "--sync-targets", str(tmp_path / "targets"), "--output", str(out), "--samples", "100"])
    assert code == 0
    report = json.loads((out / "h2.json").read_text())
    assert report["h2_success"] is True            # audio carries signal the visual path lacks
    dfdc = report["targets"]["dfdc"]
    assert dfdc["clips_scored_by_every_seed"] == 159   # one clip lost its audio score in one seed
    assert dfdc["estimate"] > 0
    assert set(report["holm"]) == {"test-identity-strict", "dfdc", "lavdf-test"}
    assert (out / "fusion-hifiavdf-seed43-predictions.csv").exists()
