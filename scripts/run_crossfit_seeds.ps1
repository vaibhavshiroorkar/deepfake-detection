<#
.SYNOPSIS
Train nested source-disjoint folds and fit one fusion head per seed.
.DESCRIPTION
Use existing split and cache inputs. Write corrected evidence to a separate run.
Completion receipts bind each command to its input and output digests.
#>
[CmdletBinding()]
param(
    [int[]]$Seeds = @(17, 29, 43),
    [int]$Folds = 3,
    [int]$MinFreeCommitGB = 12,
    [string]$SourceRun = "runs/program-20260906",
    [string]$TrainingManifest = "",
    [string]$CacheIndex = "",
    [string]$CacheRoot = "",
    [string]$CacheAudit = "",
    [string]$OutputRun = "runs/nested-crossfit-v2-20261024"
)
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root
$python = ".venv/Scripts/python.exe"
$audit = if ($CacheAudit) { $CacheAudit } else { "$SourceRun/cache-audit.json" }
$cacheRoot = if ($CacheRoot) { $CacheRoot } else { "$SourceRun/cache" }
$prepHash = (Get-Content -LiteralPath $audit -Raw | ConvertFrom-Json).preprocessing_hash
if (-not $prepHash) { throw "No preprocessing hash in $audit" }
$splitHash = Split-Path -Leaf $SourceRun
$cacheIndex = if ($CacheIndex) { $CacheIndex } else { "$SourceRun/cache-index.csv" }
$manifest = if ($TrainingManifest) { $TrainingManifest } else { "$SourceRun/split/train-usable.csv" }

function Invoke-VerifiedStage {
    param([string]$Name, [string[]]$Outputs, [string[]]$Inputs, [string[]]$Command)
    $os = Get-CimInstance Win32_OperatingSystem
    if ($os.FreeVirtualMemory / 1MB -lt $MinFreeCommitGB) {
        throw "Insufficient free commit memory before $Name"
    }
    $stageArgs = @("scripts/research_artifacts.py", "--receipt", "$target/receipts/$Name.json",
                   "--outputs") + $Outputs + @("--inputs") + $Inputs + @("--") + $Command
    & $python @stageArgs 2>&1 | Tee-Object -FilePath "$target/logs/$Name.log"
    if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE" }
}

foreach ($seed in $Seeds) {
    $target = "$OutputRun/seed$seed"
    New-Item -ItemType Directory -Force -Path "$target/logs" | Out-Null
    $foldOutputs = @("$target/folds/crossfit-audit.json")
    for ($fold = 0; $fold -lt $Folds; $fold++) {
        foreach ($part in @("train", "validation", "holdout")) {
            $foldOutputs += "$target/folds/fold-$fold-$part.csv"
        }
    }
    Invoke-VerifiedStage -Name "split" -Outputs $foldOutputs -Inputs @($manifest) -Command @(
        $python, "-m", "deepfake_detection.cli", "split", "crossfit",
        "--manifest", $manifest, "--output-dir", "$target/folds", "--dataset", "FakeAVCeleb",
        "--folds", "$Folds", "--seed", "$seed", "--validation-fraction", "0.15"
    )
    $stores = @()
    for ($fold = 0; $fold -lt $Folds; $fold++) {
        foreach ($branch in @("visual", "audio", "sync")) {
            foreach ($part in @("train", "validation", "holdout")) {
                $source = "$target/folds/fold-$fold-$part.csv"
                $usable = "$target/folds/fold-$fold-$part-$branch.csv"
                $usableAudit = "$usable-audit.json"
                Invoke-VerifiedStage -Name "usable-$fold-$part-$branch" -Outputs @($usable, $usableAudit) -Inputs @($source, $cacheIndex, $audit) -Command @(
                    $python, "-m", "deepfake_detection.cli", "manifest", "usable",
                    "--manifest", $source, "--output", $usable, "--audit", $usableAudit,
                    "--cache-index", $cacheIndex, "--cache-root", $cacheRoot,
                    "--dataset", "FakeAVCeleb", "--branch", $branch, "--preprocessing-hash", $prepHash
                )
            }
            $checkpoint = "$target/checkpoints/fold$fold-$branch.pt"
            $history = "$target/checkpoints/fold$fold-$branch-history.json"
            $train = "$target/folds/fold-$fold-train-$branch.csv"
            $validation = "$target/folds/fold-$fold-validation-$branch.csv"
            $extra = if ($branch -eq "sync") {
                @("--batch-size", "4", "--accumulation-steps", "4", "--heads-epochs", "1")
            } else { @("--batch-size", "8", "--accumulation-steps", "2", "--freeze-epochs", "1") }
            $command = @($python, "-m", "deepfake_detection.cli", "train", $branch,
                "--train-manifest", $train, "--validation-manifest", $validation,
                "--cache-index", $cacheIndex, "--cache-root", $cacheRoot, "--dataset", "FakeAVCeleb",
                "--checkpoint", $checkpoint, "--history", $history, "--run-id", "nested-fold$fold-$branch-seed$seed",
                "--split-hash", $splitHash, "--preprocessing-hash", $prepHash,
                "--device", "cuda", "--epochs", "4", "--learning-rate", "0.0001", "--weight-decay", "0.0001",
                "--patience", "2", "--workers", "0", "--seed", "$seed") + $extra
            Invoke-VerifiedStage -Name "train-$fold-$branch" -Outputs @($checkpoint, $history) -Inputs @($train, $validation, $cacheIndex, $audit) -Command $command
        }
        $store = "$target/features/fold-$fold.parquet"
        $report = "$target/features/fold-$fold-report.json"
        $holdout = "$target/folds/fold-$fold-holdout-visual.csv"
        $checkpoints = @("$target/checkpoints/fold$fold-visual.pt", "$target/checkpoints/fold$fold-audio.pt", "$target/checkpoints/fold$fold-sync.pt")
        Invoke-VerifiedStage -Name "export-$fold" -Outputs @($store, $report) -Inputs (@($holdout, $cacheIndex, $audit) + $checkpoints) -Command @(
            $python, "-m", "deepfake_detection.cli", "features", "export", "--manifest", $holdout,
            "--cache-index", $cacheIndex, "--cache-root", $cacheRoot, "--feature-store", $store,
            "--report", $report, "--allow-unavailable", "--dataset", "FakeAVCeleb", "--run-id", "nested-fold$fold-seed$seed", "--partition-role", "oof",
            "--visual-checkpoint", $checkpoints[0], "--audio-checkpoint", $checkpoints[1], "--sync-checkpoint", $checkpoints[2], "--device", "cuda"
        )
        $stores += $store
    }
    $combined = "$target/features/store.parquet"
    $merge = 'import sys; from pathlib import Path; from deepfake_detection.fusion.store import FeatureStore; FeatureStore(Path(sys.argv[1])).write([row for name in sys.argv[2:] for row in FeatureStore(Path(name)).read()])'
    Invoke-VerifiedStage -Name "merge" -Outputs @($combined) -Inputs $stores -Command (@($python, "-c", $merge, $combined) + $stores)
    Invoke-VerifiedStage -Name "fusion" -Outputs @("$target/checkpoints/fusion-logistic.joblib", "$target/checkpoints/fusion-logistic.json") -Inputs @($combined) -Command @(
        $python, "-m", "deepfake_detection.cli", "train", "fusion", "--feature-store", $combined,
        "--output", "$target/checkpoints/fusion-logistic.joblib", "--metadata", "$target/checkpoints/fusion-logistic.json",
        "--model", "logistic", "--branches", "visual", "audio", "sync", "--drop-incomplete"
    )
}
