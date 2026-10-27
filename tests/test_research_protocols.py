import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(SCRIPTS))
    return module


def test_strict_ffpp_probe_removes_donor_overlap():
    probe = script("train_logo_probe")
    names = np.array(
        [
            "real/000.mp4",
            "real/001.mp4",
            "real/002.mp4",
            "fake/a/000_001.mp4",
            "fake/a/001_002.mp4",
            "fake/b/002_000.mp4",
        ]
    )
    groups = np.array(["real", "real", "real", "a", "a", "b"])
    sources, _, _, masks = probe.split_masks(
        groups,
        names,
        seed=17,
        test_fraction=0.34,
        identity_protocol="ffpp-strict",
    )
    for train, test in masks.values():

        def identities(mask):
            return {
                identity
                for name in names[mask]
                for identity in Path(name).stem.split("_")
            }

        assert not identities(train) & identities(test)
    assert len(sources) == len(names)


def test_probe_records_scores_arguments_and_preserves_evidence(tmp_path):
    probe = script("train_logo_probe")
    paths, groups, labels = [], [], []
    for group in ("real", "a", "b"):
        for i in range(12):
            paths.append(f"{group}/{i:03d}.mp4")
            groups.append(group)
            labels.append(int(group != "real"))
    features = tmp_path / "features.npz"
    np.savez(
        features,
        features=np.arange(72).reshape(36, 2),
        labels=labels,
        groups=groups,
        files=paths,
    )
    output = tmp_path / "probe.json"
    args = [
        "--features",
        str(features),
        "--output",
        str(output),
        "--bootstrap-samples",
        "5",
        "--seed",
        "29",
        "--regularization",
        "0.5",
    ]
    assert probe.main(args) == 0
    report = json.loads(output.read_text())
    assert report["seed"] == 29
    assert report["regularization"] == 0.5
    assert len(report["feature_sha256"]) == 64
    assert report["source_assignments"]
    predictions = Path(report["predictions"])
    assert predictions.is_file()
    assert len(report["predictions_sha256"]) == 64
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        probe.main(args)
    assert output.read_bytes() == before


def test_recipe_failure_returns_nonzero(monkeypatch, tmp_path):
    recipes = script("run_visual_recipes")
    monkeypatch.setattr(recipes, "prepare", lambda _: "prep")
    monkeypatch.setattr(recipes, "train", lambda *args: False)
    assert recipes.main(["--run-dir", str(tmp_path), "--arms", "plain"]) == 1


def test_recipe_rejects_unverified_existing_checkpoint(tmp_path):
    recipes = script("run_visual_recipes")
    checkpoint = tmp_path / "checkpoints" / "plain-seed17.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"partial")
    with pytest.raises(ValueError, match="unverified|receipt"):
        recipes.train(tmp_path, "plain", 17, "prep", 8)


def test_registry_does_not_accept_declared_seed_count(tmp_path):
    registry = script("update_result_registry")
    report = tmp_path / "report.json"
    report.write_text("{}")
    result = registry.Result("test", "results", "run", report, None, seeds=3)
    assert registry._status(result, True) != "accepted"


@pytest.mark.parametrize(
    "result_id",
    ["C-df26-logo-pe", "A-encoder-comparison", "B-lavdf-lipsync", "A-fusion-in-domain"],
)
def test_registry_rejects_historical_evidence_with_known_protocol_defects(result_id):
    registry = script("update_result_registry")
    result = next(row for row in registry.RESULTS if row.result_id == result_id)
    assert registry._status(result, True) == "requires rerun"


def test_feature_resume_rejects_legacy_before_loading_encoder(tmp_path):
    extractor = script("extract_probe_features")
    features = tmp_path / "features.npz"
    features.write_bytes(b"historical features")
    features.with_suffix(".json").write_text('{"encoder": "pe"}')
    with pytest.raises(ValueError, match="normalisation|legacy"):
        extractor.main(
            ["--folder", str(tmp_path), "--output", str(features), "--resume"]
        )
    assert features.read_bytes() == b"historical features"


def test_completion_receipt_rejects_changed_config_or_output(tmp_path):
    artifacts = script("research_artifacts")
    output = tmp_path / "checkpoint"
    output.write_bytes(b"complete")
    receipt = tmp_path / "receipt.json"
    artifacts.record_completion(receipt, {"seed": 17}, [output])
    assert artifacts.completed(receipt, {"seed": 17}, [output])
    with pytest.raises(ValueError, match="mismatch"):
        artifacts.completed(receipt, {"seed": 29}, [output])
    output.write_bytes(b"changed")
    with pytest.raises(ValueError, match="mismatch"):
        artifacts.completed(receipt, {"seed": 17}, [output])


def test_recipe_evaluation_failure_returns_nonzero(monkeypatch, tmp_path):
    recipes = script("run_visual_recipes")
    monkeypatch.setattr(recipes, "prepare", lambda _: "prep")
    monkeypatch.setattr(recipes, "train", lambda *args: True)
    monkeypatch.setattr(recipes, "evaluate", lambda *args: False)
    assert recipes.main(["--run-dir", str(tmp_path), "--arms", "plain"]) == 1


def test_stage_receipt_requires_success_and_unchanged_inputs(tmp_path):
    artifacts = script("research_artifacts")
    source = tmp_path / "manifest.csv"
    source.write_text("original")
    output = tmp_path / "checkpoint"
    receipt = tmp_path / "stage.json"
    command = [
        sys.executable,
        "-c",
        "from pathlib import Path; Path(__import__('sys').argv[1]).write_text('trained')",
        str(output),
    ]
    assert artifacts.run_stage(receipt, command, [output], [source]) == 0
    assert artifacts.run_stage(receipt, command, [output], [source]) == 0
    source.write_text("changed")
    with pytest.raises(ValueError, match="mismatch"):
        artifacts.run_stage(receipt, command, [output], [source])
    failed = tmp_path / "failed.json"
    assert (
        artifacts.run_stage(
            failed,
            [sys.executable, "-c", "raise SystemExit(7)"],
            [tmp_path / "missing"],
            [],
        )
        == 7
    )
    assert not failed.exists()


def test_stage_receipt_rejects_changed_cache_payload_at_the_same_path(tmp_path):
    artifacts = script("research_artifacts")
    cached = tmp_path / "clip.npz"
    cached.write_bytes(b"old audio lengths")
    index = tmp_path / "cache-index.csv"
    index.write_text("clip_id,cache_path\nclip,clip.npz\n")
    output = tmp_path / "checkpoint"
    command = [
        sys.executable,
        "-c",
        "from pathlib import Path; Path(__import__('sys').argv[1]).write_text('trained')",
        str(output),
    ]
    receipt = tmp_path / "stage.json"
    artifacts.run_stage(receipt, command, [output], [index])
    cached.write_bytes(b"new audio lengths")
    with pytest.raises(ValueError, match="mismatch"):
        artifacts.run_stage(receipt, command, [output], [index])


def test_program_receipts_distinguish_prediction_inputs_from_outputs(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("Windows PowerShell is required for this runner")
    predictions = tmp_path / "predictions.csv"
    predictions.write_text(
        "label,probability,source,method\n0,0.1,real,real\n1,0.9,fake,fake\n"
    )
    wrapper = tmp_path / "invoke.ps1"
    wrapper.write_text(
        r"""param([string]$Root, [string]$Run, [string]$Python)
Set-Location -LiteralPath $Root
$run = $Run
$logs = "$Run/logs"
$python = $Python
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$ast = [System.Management.Automation.Language.Parser]::ParseFile("$Root/scripts/run_program.ps1", [ref]$null, [ref]$null)
foreach ($name in @("Write-Stage", "Invoke-Ddf")) {
    $node = $ast.Find({ param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $name }, $true)
    Invoke-Expression $node.Extent.Text
}
Invoke-Ddf "metrics" evaluate predictions --predictions "$run/predictions.csv" --output "$run/metrics.json" --bootstrap-samples 5 --threshold 0.5 --seed 17
""",
        encoding="utf-8",
    )
    command = [
        powershell,
        "-NoProfile",
        "-File",
        str(wrapper),
        "-Root",
        str(SCRIPTS.parent),
        "-Run",
        str(tmp_path),
        "-Python",
        sys.executable,
    ]
    # Only the local test wrapper is executed. No shell reads the data paths.
    first = subprocess.run(command, capture_output=True, text=True, check=False)  # noqa: S603
    assert first.returncode == 0, first.stdout + first.stderr
    assert (tmp_path / "metrics.json").is_file()
    second = subprocess.run(command, capture_output=True, text=True, check=False)  # noqa: S603
    assert second.returncode == 0, second.stdout + second.stderr
    assert "Verified completed stage" in second.stdout
    predictions.write_text(predictions.read_text().replace("0.9", "0.8"))
    changed = subprocess.run(command, capture_output=True, text=True, check=False)  # noqa: S603
    assert changed.returncode != 0
    assert "mismatch" in changed.stdout + changed.stderr
