"""build_numbers.py reads H2 and the branch metrics into numbers.json keys."""

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_numbers.py"


def _module(tmp_path: Path):
    spec = importlib.util.spec_from_file_location("build_numbers", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT, module.RESULTS = tmp_path, tmp_path / "results"
    return module


def test_h2_keys(tmp_path):
    module = _module(tmp_path)
    out = tmp_path / "results" / "p001-h2"
    out.mkdir(parents=True)
    target = {"estimate": 0.0213, "lower": -0.004, "upper": 0.051, "p_one_sided": 0.061,
              "clips": 2900, "identities": 190, "per_seed": [0.02, 0.025, 0.019]}
    (out / "h2.json").write_text(json.dumps({
        "seeds": [17, 29, 43],
        "targets": {"dfdc": target},
        "holm": {"dfdc": {"p_one_sided": 0.061, "rejected_at_0.05": False}},
        "h2_success": False,
        "weights": {"17": {"branches": ["visual", "audio", "sync"], "coef": [1.2, 0.3, 0.4]}},
    }))
    numbers: dict = {}
    module.h2(numbers)
    assert numbers["h2.dfdc.estimate"]["value"] == 0.021
    assert numbers["h2.dfdc.seed29"]["value"] == 0.025
    assert numbers["h2.dfdc.holm"]["value"] == "no"
    assert numbers["h2.success"]["value"] == "no"
    assert numbers["h2.weights.seed17.sync"]["value"] == 0.4
    assert numbers["h2.dfdc.p"]["source"] == "results/p001-h2/h2.json"


def test_branch_keys_split_branch_seed_and_part(tmp_path):
    module = _module(tmp_path)
    out = tmp_path / "results" / "p001-branches"
    out.mkdir(parents=True)
    report = {"overall": {"metrics": {"roc_auc": 0.91234, "fpr": 0.2, "balanced_accuracy": 0.8}, "total": 500, "scored": 480},
              "confidence_intervals": {"roc_auc": {"lower": 0.88, "upper": 0.94}}}
    (out / "visual-seed17-test-identity-strict-metrics.json").write_text(json.dumps(report))
    numbers: dict = {}
    module.branches(numbers)
    assert numbers["visual.test-identity-strict.seed17.auc"]["value"] == 0.912
    assert numbers["visual.test-identity-strict.seed17.n"]["value"] == 500
    assert numbers["visual.test-identity-strict.seed17.scored"]["value"] == 480


def test_single_class_set_records_counts_and_no_auc(tmp_path, capsys):
    # Every DFDC clip in our subset has real audio, so the audio path's DFDC set has one class and
    # the evaluator writes metrics: null. That is a result to record, not a crash.
    module = _module(tmp_path)
    out = tmp_path / "results" / "p001-branches"
    out.mkdir(parents=True)
    report = {"overall": {"metrics": None, "total": 3032, "scored": 3032},
              "confidence_intervals": {}}
    (out / "audio-seed17-dfdc-metrics.json").write_text(json.dumps(report))
    numbers: dict = {}
    module.branches(numbers)
    assert "audio.dfdc.seed17.auc" not in numbers
    assert numbers["audio.dfdc.seed17.n"]["value"] == 3032
    assert numbers["audio.dfdc.seed17.single_class"]["value"] == "yes"
    assert "one class" in capsys.readouterr().out


def test_h2_levels_give_each_path_auc_on_the_clip_label(tmp_path):
    # The H2 differences hide the levels: +0.17 over a visual path below chance reads differently
    # from +0.17 over 0.9. Each path's AUC per seed, and its range over seeds, come from the predictions.
    module = _module(tmp_path)
    out = tmp_path / "results" / "p001-h2"
    out.mkdir(parents=True)
    rows = "clip_id,label,source,fusion,visual_logit,audio_logit,sync_logit\n" \
           "a,1,s,0.9,-1,2,0.5\nb,1,s,0.8,1,3,0.1\nc,0,s,0.1,2,1,0.3\nd,0,s,0.2,0,0,0.2\n"
    for seed in (17, 29):
        (out / f"fusion-dfdc-seed{seed}-predictions.csv").write_text(rows)
    numbers: dict = {}
    module.h2_levels(numbers)
    assert numbers["h2.dfdc.seed17.auc.fusion"]["value"] == 1.0
    assert numbers["h2.dfdc.seed17.auc.visual"]["value"] == 0.25
    assert numbers["h2.dfdc.seed29.auc.audio"]["value"] == 1.0
    assert numbers["h2.dfdc.auc.visual.min"]["value"] == 0.25
    assert numbers["h2.dfdc.auc.sync.max"]["value"] == 0.5


def test_h2_checks_split_the_gains_by_manipulation_type(tmp_path):
    # Post-hoc checks on H2: the FakeAVCeleb gain without the cloned-voice clips, and the visual
    # path on each LAV-DF fake type against the reals.
    module = _module(tmp_path)
    h2 = tmp_path / "results" / "p001-h2"
    branches = tmp_path / "results" / "p001-branches"
    h2.mkdir(parents=True)
    branches.mkdir(parents=True)
    (h2 / "fusion-test-identity-strict-seed17-predictions.csv").write_text(
        "clip_id,label,source,fusion,visual_logit,audio_logit,sync_logit\n"
        "r1,0,s,0.1,0,0,0\nr2,0,s,0.2,1,0,0\nf1,1,s,0.9,2,0,0\nf2,1,s,0.8,3,0,0\nc1,1,s,0.7,-1,0,0\n")
    types = "clip_id,label,probability,predicted,source,manipulation_type,method\n"
    (branches / "visual-seed17-test-identity-strict-predictions.csv").write_text(
        types + "r1,0,0.1,0,s,RealVideo-RealAudio,real\nr2,0,0.2,0,s,RealVideo-RealAudio,real\n"
        "f1,1,0.9,1,s,FakeVideo-RealAudio,fsgan\nf2,1,0.8,1,s,FakeVideo-FakeAudio,wav2lip\n"
        "c1,0,0.1,0,s,RealVideo-FakeAudio,rtvc\n")
    (branches / "visual-seed17-lavdf-test-predictions.csv").write_text(
        types + "a,0,0.6,1,s,RealVideo-RealAudio,x\nb,0,0.8,1,s,RealVideo-RealAudio,x\n"
        "c,1,0.1,0,s,RealVideo-FakeAudio,x\nd,1,0.7,1,s,FakeVideo-RealAudio,x\n")
    numbers: dict = {}
    module.h2_checks(numbers)
    # All clips: fusion 1.0, visual 4/6 (c1 ranks below both reals); without c1 both are 1.0.
    assert numbers["h2.test-identity-strict.seed17.diff_all"]["value"] == 0.333
    assert numbers["h2.test-identity-strict.seed17.diff_without_cloned_voice"]["value"] == 0.0
    assert numbers["h2.test-identity-strict.without_cloned_voice.max"]["value"] == 0.0
    assert numbers["visual.lavdf-test.seed17.type.RealVideo-FakeAudio"]["value"] == 0.0
    assert numbers["visual.lavdf-test.seed17.type.FakeVideo-RealAudio"]["value"] == 0.5
    assert numbers["visual.lavdf-test.type.RealVideo-FakeAudio.max"]["value"] == 0.0
    assert numbers["visual.lavdf-test.type.max"]["value"] == 0.5
