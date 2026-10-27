<#
.SYNOPSIS
Run corrected caches, nested cross-fitting, final branches and evaluation.
.DESCRIPTION
Historical splits and reports remain inputs. New caches, usable manifests,
checkpoints, feature stores and evaluation reports use separate v2 paths.
Every stage verifies its completion receipt before reuse. A child failure stops
its dependents. Training stays sequential to fit this host's memory budget.
.PARAMETER SkipCache
Reuse the original source cache explicitly. Audio training still requires valid
sample counts; rebuild caches that predate those counts before masked training.
#>
[CmdletBinding()]
param(
    [int]$Folds = 3,
    [int]$Shards = 3,
    [switch]$SkipCache
)

# Not "Stop": Windows PowerShell turns a native command's stderr into a
# terminating error, and python reports progress there. Exit codes are checked.
$ErrorActionPreference = "Continue"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

$run = "runs/program-20260906"
$cacheRoot = if ($SkipCache) { "$run/cache" } else { "$run/cache-v2" }
$cacheIndex = if ($SkipCache) { "$run/cache-index.csv" } else { "$run/cache-index-v2.csv" }
$cacheAudit = if ($SkipCache) { "$run/cache-audit.json" } else { "$run/cache-audit-v2.json" }
$python = ".venv/Scripts/python.exe"
$logs = "$run/logs"
# Read from this run's own cache audit rather than hardcoded. The hash covers
# `code_version`, so a literal copied from another run's audit rejects every
# clip as a preprocessing mismatch, and `manifest usable` then reports that no
# clip has any view at all. Hardcoding it cost a restart.
$auditPath = if (Test-Path $cacheAudit) { $cacheAudit } else { "$run/cache-audit.json" }
if (-not (Test-Path $auditPath)) { throw "No cache audit at $auditPath; build the cache first." }
$prepHash = (Get-Content $auditPath -Raw | ConvertFrom-Json).preprocessing_hash
if (-not $prepHash) { throw "Cache audit at $auditPath records no preprocessing hash." }
foreach ($d in @($logs, "$run/checkpoints-v2", "$run/evaluation-v2", "$run/features-v2", "$run/folds", "$run/split-v2")) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
}
$transcript = "$logs/program.log"

function Write-Stage {
    param([string]$Message)
    $line = "=== $((Get-Date).ToString('s'))  $Message"
    Write-Host $line
    Add-Content -Path $transcript -Value $line -Encoding utf8
}

function Invoke-Ddf {
    param([string]$Name, [Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    Write-Stage "$Name start"
    $started = Get-Date
    $outputs = @()
    $inputs = @()
    $verb = "$($Arguments[0]) $($Arguments[1])"
    $outputFlags = @("--output", "--index", "--audit", "--report")
    $inputFlags = @("--manifest", "--cache-index", "--checkpoint", "--fusion-model",
                    "--visual-checkpoint", "--audio-checkpoint", "--sync-checkpoint")
    if ($verb -eq "evaluate branch") { $outputFlags += "--predictions" }
    else { $inputFlags += "--predictions" }
    if ($verb -eq "features export") { $outputFlags += "--feature-store" }
    else { $inputFlags += "--feature-store" }
    for ($i = 0; $i -lt $Arguments.Count - 1; $i++) {
        if ($Arguments[$i] -in $outputFlags) { $outputs += $Arguments[$i + 1] }
        if ($Arguments[$i] -in $inputFlags) { $inputs += $Arguments[$i + 1] }
        if ($Arguments[$i] -in @("--indexes", "--audits")) {
            for ($j = $i + 1; $j -lt $Arguments.Count -and -not $Arguments[$j].StartsWith("--"); $j++) {
                $inputs += $Arguments[$j]
            }
        }
    }
    if (-not $outputs.Count) { throw "No declared outputs for $Name" }
    $stageArgs = @("scripts/research_artifacts.py", "--receipt", "$run/receipts-v2/$Name.json",
                   "--outputs") + $outputs + @("--inputs") + $inputs + @("--", $python, "-m", "deepfake_detection.cli") + $Arguments
    & $python @stageArgs 2>&1 | Tee-Object -FilePath "$logs/$Name.log"
    $code = $LASTEXITCODE
    Write-Stage "$Name exit=$code elapsed=$([int]((Get-Date)-$started).TotalSeconds)s"
    if ($code -ne 0) { throw "$Name failed with exit code $code" }
    return $code
}

function Build-Cache {
    param([string]$Name, [string]$Manifest, [string]$DatasetName)
    Write-Stage "cache-$Name start ($Shards shards)"
    $started = Get-Date
    $jobs = @()
    for ($i = 0; $i -lt $Shards; $i++) {
        $jobs += Start-Process -FilePath $python -PassThru -WindowStyle Hidden `
            -RedirectStandardOutput "$logs/cache-$Name-$i.log" `
            -RedirectStandardError "$logs/cache-$Name-$i.err" `
            -ArgumentList @("scripts/research_artifacts.py", "--receipt", "$run/receipts-v2/cache-$Name-$i.json",
                "--outputs", "$run/cache-index-v2-$Name-$i.csv", "$run/cache-audit-v2-$Name-$i.json", "--inputs", $Manifest,
                "--", $python, "-m","deepfake_detection.cli","cache","build",
                "--manifest",$Manifest,"--dataset-root","data",
                "--cache-root",$cacheRoot,"--index","$run/cache-index-v2-$Name-$i.csv",
                "--audit","$run/cache-audit-v2-$Name-$i.json","--dataset",$DatasetName,
                "--device","cuda","--code-version","program-v1","--skip-cached",
                "--shard","$i/$Shards")
    }
    foreach ($job in $jobs) { $null = $job.Handle }
    $jobs | Wait-Process
    foreach ($job in $jobs) {
        $job.Refresh()
        if ($job.ExitCode -ne 0) { throw "Cache shard $($job.Id) failed with exit code $($job.ExitCode)" }
    }
    Write-Stage "cache-$Name shards done elapsed=$([int]((Get-Date)-$started).TotalSeconds)s"
    $idx = (0..($Shards-1) | ForEach-Object { "$run/cache-index-v2-$Name-$_.csv" })
    $aud = (0..($Shards-1) | ForEach-Object { "$run/cache-audit-v2-$Name-$_.json" })
    Invoke-Ddf "cache-merge-$Name" cache merge --indexes @idx --audits @aud `
        --index "$run/cache-index-v2-$Name.csv" --audit "$run/cache-audit-v2-$Name.json" | Out-Null
}

function Train-Branch {
    param([string]$Branch, [string]$Name, [string]$TrainManifest,
          [string]$ValManifest, [int]$Epochs, [int]$Seed = 17)
    $checkpoint = "$run/checkpoints-v2/$Name.pt"
    $history = "$run/checkpoints-v2/$Name-history.json"
    $extra = if ($Branch -eq "sync") {
        @("--batch-size", "4", "--accumulation-steps", "4", "--heads-epochs", "1")
    } else { @("--batch-size", "8", "--accumulation-steps", "2", "--freeze-epochs", "1") }
    $command = @($python, "-m", "deepfake_detection.cli", "train", $Branch,
        "--train-manifest", $TrainManifest, "--validation-manifest", $ValManifest,
        "--cache-index", $cacheIndex, "--cache-root", $cacheRoot,
        "--dataset", "FakeAVCeleb", "--checkpoint", $checkpoint, "--history", $history,
        "--run-id", $Name, "--split-hash", "program-20260906", "--preprocessing-hash", $prepHash,
        "--device", "cuda", "--epochs", "$Epochs", "--learning-rate", "0.0001", "--weight-decay", "0.0001",
        "--patience", "2", "--workers", "0", "--seed", "$Seed") + $extra
    $stageArgs = @("scripts/research_artifacts.py", "--receipt", "$run/receipts-v2/train-$Name.json",
                   "--outputs", $checkpoint, $history, "--inputs", $TrainManifest, $ValManifest,
                   $cacheIndex, $cacheAudit, "--") + $command
    & $python @stageArgs 2>&1 | Tee-Object -FilePath "$logs/train-$Name.log"
    if ($LASTEXITCODE -ne 0) { throw "train-$Name failed with exit code $LASTEXITCODE" }
    return $true
}

function Usable-Manifest {
    param([string]$Source, [string]$Target, [string]$Branch, [string]$DatasetName = "FakeAVCeleb")
    Invoke-Ddf "usable-$([System.IO.Path]::GetFileNameWithoutExtension($Target))" manifest usable `
        --manifest $Source --cache-index $cacheIndex --cache-root $cacheRoot `
        --output $Target --audit "$run/usable-v2-$([System.IO.Path]::GetFileNameWithoutExtension($Target)).json" `
        --dataset $DatasetName --branch $Branch --preprocessing-hash $prepHash | Out-Null
}


function Score-Fusion {
    <#
    Score the fusion model on a partition it never trained on.

    Three steps, because fusion does not read clips: the frozen branches turn
    each clip into logits and quality features, the fusion artifact turns those
    into one probability, and `evaluate predictions` turns the probabilities
    into metrics with bootstrap intervals.

    Each partition gets its own feature store. `train fusion` rejects a store
    holding anything but out-of-fold rows, so test and external features must
    not be appended to the store the fusion model was fitted on.
    #>
    param([string]$Name, [string]$Manifest, [string]$DatasetName, [string]$Role)
    if (-not (Test-Path "$run/nested-crossfit-v2/seed17/checkpoints/fusion-logistic.joblib")) {
        Write-Stage "score-fusion-$Name skipped, no fusion model"
        return
    }
    foreach ($b in @("visual","audio","sync")) {
        if (-not (Test-Path "$run/checkpoints-v2/final-$b-seed17.pt")) {
            Write-Stage "score-fusion-$Name skipped, final-$b-seed17 is missing"
            return
        }
    }
    $store = "$run/features-v2/$Name.parquet"
    & {
        Invoke-Ddf "features-$Name" features export `
            --manifest $Manifest --cache-index $cacheIndex `
            --cache-root $cacheRoot --feature-store $store `
            --report "$run/features-v2/$Name-report.json" --allow-unavailable `
            --dataset $DatasetName --run-id "final" --partition-role $Role `
            --visual-checkpoint "$run/checkpoints-v2/final-visual-seed17.pt" `
            --audio-checkpoint "$run/checkpoints-v2/final-audio-seed17.pt" `
            --sync-checkpoint "$run/checkpoints-v2/final-sync-seed17.pt" `
            --device cuda | Out-Null
    }
    if (-not (Test-Path $store)) {
        Write-Stage "score-fusion-$Name skipped, feature export produced nothing"
        return
    }
    Invoke-Ddf "fusion-score-$Name" features score --feature-store $store `
        --fusion-model "$run/nested-crossfit-v2/seed17/checkpoints/fusion-logistic.joblib" `
        --output "$run/evaluation-v2/fusion-nested-v2-$Name-predictions.csv" | Out-Null
    Invoke-Ddf "fusion-metrics-$Name" evaluate predictions `
        --predictions "$run/evaluation-v2/fusion-nested-v2-$Name-predictions.csv" `
        --output "$run/evaluation-v2/fusion-nested-v2-$Name-metrics.json" `
        --threshold 0.5 --bootstrap-samples 1000 --seed 17 | Out-Null
}

Write-Stage "program start folds=$Folds"
try {
    # ---- 1. cache -------------------------------------------------------
    if (-not $SkipCache) {
        $union = "$run/cache-manifest-v2.csv"
        $sourceManifests = @("$run/split/train.csv", "$run/split/val.csv", "$run/split/test.csv")
        $mergeCode = 'import pandas as pd,sys; pd.concat([pd.read_csv(name) for name in sys.argv[2:]], ignore_index=True).to_csv(sys.argv[1], index=False)'
        $stageArgs = @("scripts/research_artifacts.py", "--receipt", "$run/receipts-v2/cache-manifest.json",
                       "--outputs", $union, "--inputs") + $sourceManifests + @("--", $python, "-c", $mergeCode, $union) + $sourceManifests
        & $python @stageArgs
        if ($LASTEXITCODE -ne 0) { throw "Cache manifest creation failed" }
        Build-Cache -Name "fakeavceleb" -Manifest $union -DatasetName "FakeAVCeleb"
        Build-Cache -Name "celebdf" -Manifest "runs/full-20260904/celebdf-test-manifest.csv" -DatasetName "Celeb-DF-v2"
        Build-Cache -Name "mnw" -Manifest "runs/full-20260904/mnw-manifest.csv" -DatasetName "MNW"
        Build-Cache -Name "dfdc" -Manifest "runs/streams-20260905/dfdc-manifest.csv" -DatasetName "DFDC"
    }
    if (-not $SkipCache) {
    $idx = @("$run/cache-index-v2-fakeavceleb.csv","$run/cache-index-v2-celebdf.csv",
             "$run/cache-index-v2-mnw.csv","$run/cache-index-v2-dfdc.csv")
    $aud = @("$run/cache-audit-v2-fakeavceleb.json","$run/cache-audit-v2-celebdf.json",
             "$run/cache-audit-v2-mnw.json","$run/cache-audit-v2-dfdc.json")
    Invoke-Ddf "cache-merge-all" cache merge --indexes @idx --audits @aud `
        --index $cacheIndex --audit $cacheAudit | Out-Null
    }

    $prepHash = (Get-Content -LiteralPath $cacheAudit -Raw | ConvertFrom-Json).preprocessing_hash
    if (-not $prepHash) { throw "No preprocessing hash in $cacheAudit" }
    foreach ($p in @("train","val","test")) {
        Usable-Manifest "$run/split/$p.csv" "$run/split-v2/$p-usable.csv" "visual"
        Usable-Manifest "$run/split/$p.csv" "$run/split-v2/$p-audio-usable.csv" "audio"
        Usable-Manifest "$run/split/$p.csv" "$run/split-v2/$p-sync-usable.csv" "sync"
    }

    # Inner validation selects epochs. Outer holdouts only produce OOF rows.
    $nestedRun = "$run/nested-crossfit-v2"
    & "$PSScriptRoot/run_crossfit_seeds.ps1" -SourceRun $run -CacheIndex $cacheIndex -CacheRoot $cacheRoot -CacheAudit $cacheAudit -TrainingManifest "$run/split-v2/train-usable.csv" -OutputRun $nestedRun -Seeds @(17) -Folds $Folds
    if (-not $?) { throw "Nested cross-fitting failed" }

    # ---- 4. final branch models ----------------------------------------
    foreach ($seed in @(17, 29, 43)) {
        Train-Branch "visual" "final-visual-seed$seed" "$run/split-v2/train-usable.csv" `
            "$run/split-v2/val-usable.csv" 8 $seed | Out-Null
    }
    Train-Branch "audio" "final-audio-seed17" "$run/split-v2/train-audio-usable.csv" `
        "$run/split-v2/val-audio-usable.csv" 6 | Out-Null
    Train-Branch "sync" "final-sync-seed17" "$run/split-v2/train-sync-usable.csv" `
        "$run/split-v2/val-sync-usable.csv" 6 | Out-Null

    # ---- 5. evaluation --------------------------------------------------
    $visual = "$run/checkpoints-v2/final-visual-seed17.pt"
    if (Test-Path $visual) {
        foreach ($t in @(
            @("in-domain-test", "$run/split-v2/test-usable.csv", "FakeAVCeleb", "development_test"),
            @("celebdf", "runs/full-20260904/celebdf-test-manifest.csv", "Celeb-DF-v2", "generalization_celebdf"),
            @("mnw", "runs/full-20260904/mnw-manifest.csv", "MNW", "external_mnw"),
            @("dfdc", "runs/streams-20260905/dfdc-manifest.csv", "DFDC", "generalization_celebdf")
        )) {
            Invoke-Ddf "evaluate-visual-$($t[0])" evaluate branch --branch visual `
                --checkpoint $visual --manifest $t[1] --cache-index $cacheIndex `
                --cache-root $cacheRoot --dataset $t[2] --threshold 0.5 `
                --output "$run/evaluation-v2/visual-$($t[0])-metrics.json" `
                --predictions "$run/evaluation-v2/visual-$($t[0])-predictions.csv" `
                --evidence-scope $t[3] --preprocessing-hash $prepHash | Out-Null
        }
    }
    $audio = "$run/checkpoints-v2/final-audio-seed17.pt"
    if (Test-Path $audio) {
        Invoke-Ddf "evaluate-audio-in-domain" evaluate branch --branch audio `
            --checkpoint $audio --manifest "$run/split-v2/test-usable.csv" `
            --cache-index $cacheIndex --cache-root $cacheRoot `
            --dataset FakeAVCeleb --threshold 0.5 `
            --output "$run/evaluation-v2/audio-in-domain-test-metrics.json" `
            --predictions "$run/evaluation-v2/audio-in-domain-test-predictions.csv" `
            --evidence-scope development_test --preprocessing-hash $prepHash | Out-Null
    }

    # ---- 6. fusion evaluation -----------------------------------------
    # In-domain first, then the only cross-corpus set that carries audio.
    # Celeb-DF-v2 and MNW cannot appear here: neither has an audio track, so
    # the audio and sync branches have nothing to read and fusion has no rows.
    Score-Fusion "in-domain-test" "$run/split-v2/test-usable.csv" "FakeAVCeleb" "test"
    Score-Fusion "dfdc" "runs/streams-20260905/dfdc-manifest.csv" "DFDC" "external"

    Write-Stage "program complete"
} catch {
    Write-Stage "program FAILED: $_"
    throw
}
