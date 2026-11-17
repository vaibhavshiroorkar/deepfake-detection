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


def test_a12_keys(tmp_path):
    module = _module(tmp_path)
    out = tmp_path / "results" / "p001-a12"
    out.mkdir(parents=True)
    diff = {"estimate": -0.2291, "lower": -0.274, "upper": -0.182, "per_seed": [-0.3, -0.15, -0.24]}
    (out / "a12.json").write_text(json.dumps({"targets": {"dfdc": {
        "clips": 2786,
        "auc": {"leave_one_out": [0.4, 0.5, 0.53], "visual": [0.7, 0.69, 0.71]},
        "differences": {"leave_one_out-minus-visual": diff}}}}))
    numbers: dict = {}
    module.a12(numbers)
    assert numbers["a12.dfdc.auc.leave_one_out"]["value"] == 0.477
    assert numbers["a12.dfdc.leave_one_out-minus-visual.estimate"]["value"] == -0.229
    assert numbers["a12.dfdc.leave_one_out-minus-visual.lower"]["value"] == -0.274
    assert numbers["a12.dfdc.clips"]["value"] == 2786


def test_a11_keys_and_skipped_sets(tmp_path):
    module = _module(tmp_path)
    out = tmp_path / "results" / "p001-a11"
    out.mkdir(parents=True)
    diff = {"estimate": -0.1254, "lower": -0.158, "upper": -0.091}
    (out / "compare.json").write_text(json.dumps({"results": {
        "clip-vs-visual.test-identity-strict": {"seeds": [17, 29, 43], "clips": 495,
            "auc_new": [0.86, 0.87, 0.862], "auc_old": [0.989, 0.988, 0.99], "difference": diff},
        "xlsr-vs-audio.dfdc": {"skipped": "one class in the set"}}}))
    numbers: dict = {}
    module.a11(numbers)
    assert numbers["a11.clip-vs-visual.test-identity-strict.auc_new"]["value"] == 0.864
    assert numbers["a11.clip-vs-visual.test-identity-strict.estimate"]["value"] == -0.125
    assert numbers["a11.clip-vs-visual.test-identity-strict.seeds"]["value"] == 3
    assert numbers["a11.xlsr-vs-audio.dfdc.skipped"]["value"] == "one class in the set"
    assert "a11.xlsr-vs-audio.dfdc.estimate" not in numbers


def test_a13_keys(tmp_path):
    module = _module(tmp_path)
    out = tmp_path / "results" / "p001-a13"
    out.mkdir(parents=True)
    stat = {"auc": 0.4543, "interval": [0.442, 0.464], "spearman_with_visual": 0.0123, "candidate": False}
    (out / "a13.json").write_text(json.dumps({
        "lavdf_unedited_video_fakes_vs_reals": {"clips": 1936, "statistics": {"sharpness": stat},
                                                "all_six_cross_fit_auc": 0.5641, "visual_path_auc": 0.1962}}))
    numbers: dict = {}
    module.a13(numbers)
    assert numbers["a13.lavdf.sharpness.auc"]["value"] == 0.454
    assert numbers["a13.lavdf.sharpness.low"]["value"] == 0.442
    assert numbers["a13.lavdf.sharpness.rho"]["value"] == 0.01
    assert numbers["a13.lavdf.sharpness.candidate"]["value"] == "no"
    assert numbers["a13.lavdf.all_six"]["value"] == 0.564
    assert numbers["a13.lavdf.visual"]["value"] == 0.196
    assert numbers["a13.lavdf.auc_min"]["value"] == 0.454
    assert numbers["a13.lavdf.auc_max"]["value"] == 0.454


def test_a14_keys_reuse_the_metrics_reader(tmp_path):
    module = _module(tmp_path)
    out = tmp_path / "results" / "p001-a14"
    out.mkdir(parents=True)
    report = {"overall": {"metrics": {"roc_auc": 0.9401, "fpr": 0.1, "balanced_accuracy": 0.9}, "total": 495, "scored": 495},
              "confidence_intervals": {"roc_auc": {"lower": 0.91, "upper": 0.97}}}
    (out / "audio-seed29-test-identity-strict-both-metrics.json").write_text(json.dumps(report))
    numbers: dict = {}
    module.a14(numbers)
    assert numbers["a14.test-identity-strict.both.seed29.auc"]["value"] == 0.94
    assert numbers["a14.test-identity-strict.both.seed29.auc_low"]["value"] == 0.91
    (out / "audio-seed29-test-identity-strict-none-metrics.json").write_text(json.dumps(report))
    numbers = {}
    module.a14(numbers)
    # "none" is the unperturbed rerun, so it stays out of the perturbed range.
    assert numbers["a14.test-identity-strict.perturbed_min"]["value"] == 0.94


def _write(tmp_path, folder, name, data):
    out = tmp_path / "results" / folder
    out.mkdir(parents=True, exist_ok=True)
    (out / name).write_text(json.dumps(data))


def test_a15_keys(tmp_path):
    module = _module(tmp_path)
    stream = {"17": {"f1": 0.21, "precision": 0.188, "recall": 0.238}, "baseline": {"f1": 0.8, "precision": 0.667, "recall": 1.0}}
    label = {"clips": 3815, "identities": 999, "macro_f1_heads": {"17": 0.2004, "29": 0.3527, "43": 0.2491},
             "macro_f1_baseline": 0.8001, "difference": {"estimate": -0.5321, "lower": -0.549, "upper": -0.516, "p_one_sided": 1.0},
             "per_stream": {"video": stream, "audio": stream}}
    _write(tmp_path, "p001-a15", "a15.json", {"labels": {"window": label}})
    numbers: dict = {}
    module.a15(numbers)
    assert numbers["a15.window.clips"]["value"] == 3815
    assert numbers["a15.window.heads.min"]["value"] == 0.2
    assert numbers["a15.window.heads.max"]["value"] == 0.353
    assert numbers["a15.window.baseline"]["value"] == 0.8
    assert numbers["a15.window.estimate"]["value"] == -0.532
    assert numbers["a15.window.video.seed17.precision"]["value"] == 0.188


def test_a16_keys(tmp_path):
    module = _module(tmp_path)
    cond = {"mean_auc": 0.8991, "lower": 0.848, "upper": 0.947,
            "per_seed": {"17": {"auc": 0.898, "share_rising": 0.93}, "29": {"auc": 0.895, "share_rising": 0.95}}}
    _write(tmp_path, "p001-a16", "a16.json", {"clips": 75, "identities": 75, "conditions": {"shift 0.20 s": cond, "swap": cond}})
    numbers: dict = {}
    module.a16(numbers)
    assert numbers["a16.clips"]["value"] == 75
    assert numbers["a16.shift020.auc"]["value"] == 0.899
    assert numbers["a16.swap.low"]["value"] == 0.848
    assert numbers["a16.shift020.rising.min"]["value"] == 0.93


def test_a17_keys(tmp_path):
    module = _module(tmp_path)
    _write(tmp_path, "p001-a17", "a17.json", {"vox2_speakers": {"dev": 5994, "test": 118},
                                               "fakeavceleb_sources": {"dev": 500, "test": 0, "neither": 0},
                                               "fakeavceleb_test_partition": {"dev": {"clips": 2993}}})
    numbers: dict = {}
    module.a17(numbers)
    assert numbers["a17.sources.dev"]["value"] == 500
    assert numbers["a17.sources.test"]["value"] == 0
    assert numbers["a17.vox2.test"]["value"] == 118
    assert numbers["a17.test_clips.dev"]["value"] == 2993


def test_a19_keys(tmp_path):
    module = _module(tmp_path)
    entry = {"clips": 2908, "auc": {"17": 0.487, "29": 0.551}, "auc_sync": {"17": 0.553, "29": 0.575},
             "mean_auc": 0.5191, "lower": 0.488, "upper": 0.549,
             "minus_sync": {"estimate": -0.0461, "lower": -0.066, "upper": -0.025, "p_one_sided": 1.0}}
    _write(tmp_path, "p001-a19", "a19.json", {"sets": {"dfdc": entry}})
    numbers: dict = {}
    module.a19(numbers)
    assert numbers["a19.dfdc.auc"]["value"] == 0.519
    assert numbers["a19.dfdc.minus_sync.estimate"]["value"] == -0.046
    assert numbers["a19.dfdc.minus_sync.upper"]["value"] == -0.025
    assert numbers["a19.dfdc.reading"]["value"] == "worse"
