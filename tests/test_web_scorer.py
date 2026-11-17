"""The website scorer gives the study's recorded scores on study clips.

Runs only where the P-001 checkpoints, AV-HuBERT and the raw FakeAVCeleb clips are
present (this PC); about two minutes, most of it loading models on CPU.
"""

import csv

import pytest

from deepfake_detection.paths import PROJECT_ROOT

CLIPS = ("A__id00781__00092", "D__id00781__00092_id00476_UgdYVJ6xPYg_faceswap_id01170_wavtolip")
NEEDED = (
    PROJECT_ROOT / "runs/p001/checkpoints/visual-seed17.pt",
    PROJECT_ROOT / "runs/p001/sync/sync-seed17.pt",
    PROJECT_ROOT / "models/avhubert/base_vox_iter5.pt",
    PROJECT_ROOT / "data/manifests/fakeavceleb.csv",
)
pytestmark = pytest.mark.skipif(not all(p.exists() for p in NEEDED), reason="needs P-001 models and raw data")
# Visual scores move a little on CPU (face detection runs on the given device; project log 11-17).
TOLERANCE = {"face": 0.06, "voice": 0.002, "sync": 1e-4}
RESULT = {"face": "p001-branches/visual-seed{seed}-test", "voice": "p001-branches/audio-seed{seed}-test",
          "sync": "p001-sync/sync-test-seed{seed}"}


def _recorded(check: str, seed: int) -> dict[str, float]:
    path = PROJECT_ROOT / "results" / (RESULT[check].format(seed=seed) + "-predictions.csv")
    with open(path, encoding="utf-8", newline="") as f:
        return {r["clip_id"]: float(r["probability"]) for r in csv.DictReader(f)}


@pytest.fixture(scope="module")
def scorer():
    from deepfake_detection.webapp.scorer import Scorer

    return Scorer()


@pytest.mark.parametrize("clip_id", CLIPS)
def test_scorer_reproduces_the_recorded_study_scores(scorer, clip_id, tmp_path) -> None:
    from deepfake_detection.webapp.scorer import SEEDS

    with open(PROJECT_ROOT / "data/manifests/fakeavceleb.csv", encoding="utf-8", newline="") as f:
        raw = next(r for r in csv.DictReader(f) if r["clip_id"] == clip_id)["video_path"]
    report = scorer.score(PROJECT_ROOT / "data" / raw, tmp_path)
    for check, result in report.checks.items():
        assert result.probabilities is not None, (check, result.reason)
        for seed, got in zip(SEEDS, result.probabilities, strict=True):
            assert got == pytest.approx(_recorded(check, seed)[clip_id], abs=TOLERANCE[check]), (check, seed)
    assert len(report.trace["loudness_db"]) == 50
    assert [w["start"] for w in report.trace["windows"]] == [0.0, 1.0]
