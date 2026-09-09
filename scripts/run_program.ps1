<#
.SYNOPSIS
Run the whole pipeline: three branches, out-of-fold features, fusion, evaluation.

.DESCRIPTION
Every earlier run trained one thing. This runs the pipeline the project was
designed around, end to end, so that "the whole thing works" stops being a
claim and becomes a set of artefacts.

Stages, in order, each skipped if its output already exists:

  1. Cache the sampled FakeAVCeleb split, plus the three evaluation corpora.
  2. Cross-fit folds, and train visual, audio and sync inside each fold. This
     is the expensive part and the reason fusion has never run on real data:
     fusion may only read branch scores produced by checkpoints that never saw
     the clip, so the branches have to be trained once per fold.
  3. Export out-of-fold features and fit the fusion model on them.
  4. Train the final branches on the whole training partition. These are the
     models that get evaluated and shipped; the fold models exist only to
     produce honest features for step 3.
  5. Evaluate on the held-out test partition and on every cross-corpus set.

Sequential on purpose: every stage wants the whole GPU. Re-running is safe,
because the cache skips what it has and each stage checks for its own output
before starting.

`ddf handoff update` runs after every stage so the docs track reality.

.PARAMETER Folds
Cross-fitting folds. Three is the minimum that leaves a usable holdout.

.PARAMETER SkipCache
Assume the cache is built and go straight to training.
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
$python = ".venv/Scripts/python.exe"
$logs = "$run/logs"
# Read from this run's own cache audit rather than hardcoded. The hash covers
# `code_version`, so a literal copied from another run's audit rejects every
# clip as a preprocessing mismatch, and `manifest usable` then reports that no
# clip has any view at all. Hardcoding it cost a restart.
$auditPath = "$run/cache-audit.json"
if (-not (Test-Path $auditPath)) { throw "No cache audit at $auditPath; build the cache first." }
$prepHash = (Get-Content $auditPath -Raw | ConvertFrom-Json).preprocessing_hash
if (-not $prepHash) { throw "Cache audit at $auditPath records no preprocessing hash." }
foreach ($d in @($logs, "$run/checkpoints", "$run/evaluation", "$run/features", "$run/folds")) {
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
    & $python -m deepfake_detection.cli @Arguments 2>&1 | Tee-Object -FilePath "$logs/$Name.log"
    $code = $LASTEXITCODE
    Write-Stage "$Name exit=$code elapsed=$([int]((Get-Date)-$started).TotalSeconds)s"
    & $python -m deepfake_detection.cli handoff update | Out-Null
    if ($code -gt 2) { throw "$Name failed with exit code $code" }
    return $code
}

function Build-Cache {
    param([string]$Name, [string]$Manifest, [string]$DatasetName)
    if (Test-Path "$run/cache-index-$Name.csv") {
        Write-Stage "cache-$Name already present, skipping"
        return
    }
    Write-Stage "cache-$Name start ($Shards shards)"
    $started = Get-Date
    $jobs = @()
    for ($i = 0; $i -lt $Shards; $i++) {
        $jobs += Start-Process -FilePath $python -PassThru -NoNewWindow `
            -RedirectStandardOutput "$logs/cache-$Name-$i.log" `
            -RedirectStandardError "$logs/cache-$Name-$i.err" `
            -ArgumentList @("-m","deepfake_detection.cli","cache","build",
                "--manifest",$Manifest,"--dataset-root","data",
                "--cache-root","$run/cache","--index","$run/cache-index-$Name-$i.csv",
                "--audit","$run/cache-audit-$Name-$i.json","--dataset",$DatasetName,
                "--device","cuda","--code-version","program-v1","--skip-cached",
                "--shard","$i/$Shards")
    }
    $jobs | Wait-Process
    Write-Stage "cache-$Name shards done elapsed=$([int]((Get-Date)-$started).TotalSeconds)s"
    $idx = (0..($Shards-1) | ForEach-Object { "$run/cache-index-$Name-$_.csv" })
    $aud = (0..($Shards-1) | ForEach-Object { "$run/cache-audit-$Name-$_.json" })
    Invoke-Ddf "cache-merge-$Name" cache merge --indexes @idx --audits @aud `
        --index "$run/cache-index-$Name.csv" --audit "$run/cache-audit-$Name.json" | Out-Null
}

function Train-Branch {
    param([string]$Branch, [string]$Name, [string]$TrainManifest,
          [string]$ValManifest, [int]$Epochs, [int]$Seed = 17)
    $checkpoint = "$run/checkpoints/$Name.pt"
    if (Test-Path $checkpoint) {
        Write-Stage "train-$Name already present, skipping"
        return $true
    }
    Write-Stage "train-$Name start"
    $started = Get-Date
    $extra = @()
    # Sync trains two encoders over temporal tokens, so it needs a smaller batch
    # to stay inside 16 GB of VRAM.
    if ($Branch -eq "sync") { $extra = @("--batch-size","4","--accumulation-steps","4","--heads-epochs","1") }
    else { $extra = @("--batch-size","8","--accumulation-steps","2","--freeze-epochs","1") }

    # Loader workers are a host-RAM question, not a VRAM one, and the branches
    # are not comparable. A visual batch is 8 clips of 16 frames at 224x224,
    # which is 77 MB; an audio batch of the same size is 2 MB. Four workers each
    # prefetching those, on top of a torch import per worker process, killed
    # every visual run on a host with 10 GB free while audio and sync finished.
    $workers = if ($Branch -eq "visual") { 2 } else { 4 }

    foreach ($attempt in @($workers, 0)) {
        & $python -m deepfake_detection.cli train $Branch `
            --train-manifest $TrainManifest --validation-manifest $ValManifest `
            --cache-index "$run/cache-index.csv" --cache-root "$run/cache" `
            --dataset FakeAVCeleb --checkpoint $checkpoint `
            --history "$run/checkpoints/$Name-history.json" `
            --run-id $Name --split-hash "program-20260906" --preprocessing-hash $prepHash `
            --device cuda --epochs $Epochs --learning-rate 0.0001 --weight-decay 0.0001 `
            --patience 2 --workers $attempt --seed $Seed @extra 2>&1 |
            Tee-Object -FilePath "$logs/train-$Name.log"
        $code = $LASTEXITCODE
        if ($code -eq 0 -or $attempt -eq 0) { break }
        # A worker crash is a memory failure, not a modelling one, and single
        # process loading always fits. Slower beats losing the run.
        Write-Stage "train-$Name failed with $attempt workers, retrying single-process"
    }
    Write-Stage "train-$Name exit=$code elapsed=$([int]((Get-Date)-$started).TotalSeconds)s"
    & $python -m deepfake_detection.cli handoff update | Out-Null
    if ($code -ne 0) {
        # Recorded and skipped. A failed branch loses the stages that need it,
        # not the runs that already succeeded.
        Write-Stage "train-$Name FAILED, continuing"
        return $false
    }
    return $true
}

function Usable-Manifest {
    param([string]$Source, [string]$Target, [string]$Branch, [string]$DatasetName = "FakeAVCeleb")
    if (Test-Path $Target) { return }
    Invoke-Ddf "usable-$([System.IO.Path]::GetFileNameWithoutExtension($Target))" manifest usable `
        --manifest $Source --cache-index "$run/cache-index.csv" --cache-root "$run/cache" `
        --output $Target --audit "$run/usable-$([System.IO.Path]::GetFileNameWithoutExtension($Target)).json" `
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
    if (-not (Test-Path "$run/checkpoints/fusion-logistic.joblib")) {
        Write-Stage "score-fusion-$Name skipped, no fusion model"
        return
    }
    foreach ($b in @("visual","audio","sync")) {
        if (-not (Test-Path "$run/checkpoints/final-$b-seed17.pt")) {
            Write-Stage "score-fusion-$Name skipped, final-$b-seed17 is missing"
            return
        }
    }
    $store = "$run/features/$Name.parquet"
    if (-not (Test-Path $store)) {
        Invoke-Ddf "features-$Name" features export `
            --manifest $Manifest --cache-index "$run/cache-index.csv" `
            --cache-root "$run/cache" --feature-store $store `
            --report "$run/features/$Name-report.json" `
            --dataset $DatasetName --run-id "final" --partition-role $Role `
            --visual-checkpoint "$run/checkpoints/final-visual-seed17.pt" `
            --audio-checkpoint "$run/checkpoints/final-audio-seed17.pt" `
            --sync-checkpoint "$run/checkpoints/final-sync-seed17.pt" `
            --device cuda | Out-Null
    }
    if (-not (Test-Path $store)) {
        Write-Stage "score-fusion-$Name skipped, feature export produced nothing"
        return
    }
    Invoke-Ddf "fusion-score-$Name" features score --feature-store $store `
        --fusion-model "$run/checkpoints/fusion-logistic.joblib" `
        --output "$run/evaluation/fusion-$Name-predictions.csv" | Out-Null
    Invoke-Ddf "fusion-metrics-$Name" evaluate predictions `
        --predictions "$run/evaluation/fusion-$Name-predictions.csv" `
        --output "$run/evaluation/fusion-$Name-metrics.json" `
        --threshold 0.5 --bootstrap-samples 1000 --seed 17 | Out-Null
}

Write-Stage "program start folds=$Folds"
try {
    # ---- 1. cache -------------------------------------------------------
    if (-not $SkipCache) {
        $union = "$run/cache-manifest.csv"
        if (-not (Test-Path $union)) {
            & $python -c @"
import pandas as pd
parts = [pd.read_csv(f'$run/split/{n}.csv') for n in ('train','val','test')]
pd.concat(parts, ignore_index=True).to_csv(r'$union', index=False)
"@
        }
        Build-Cache -Name "fakeavceleb" -Manifest $union -DatasetName "FakeAVCeleb"
        Build-Cache -Name "celebdf" -Manifest "runs/full-20260904/celebdf-test-manifest.csv" -DatasetName "Celeb-DF-v2"
        Build-Cache -Name "mnw" -Manifest "runs/full-20260904/mnw-manifest.csv" -DatasetName "MNW"
        Build-Cache -Name "dfdc" -Manifest "runs/streams-20260905/dfdc-manifest.csv" -DatasetName "DFDC"
    }
    $idx = @("$run/cache-index-fakeavceleb.csv","$run/cache-index-celebdf.csv",
             "$run/cache-index-mnw.csv","$run/cache-index-dfdc.csv")
    $aud = @("$run/cache-audit-fakeavceleb.json","$run/cache-audit-celebdf.json",
             "$run/cache-audit-mnw.json","$run/cache-audit-dfdc.json")
    Invoke-Ddf "cache-merge-all" cache merge --indexes @idx --audits @aud `
        --index "$run/cache-index.csv" --audit "$run/cache-audit.json" | Out-Null

    foreach ($p in @("train","val","test")) {
        Usable-Manifest "$run/split/$p.csv" "$run/split/$p-usable.csv" "visual"
        Usable-Manifest "$run/split/$p.csv" "$run/split/$p-audio-usable.csv" "audio"
        Usable-Manifest "$run/split/$p.csv" "$run/split/$p-sync-usable.csv" "sync"
    }

    # ---- 2. cross-fitting for out-of-fold features ----------------------
    Invoke-Ddf "split-crossfit" split crossfit --manifest "$run/split/train-usable.csv" `
        --output-dir "$run/folds" --dataset FakeAVCeleb --folds $Folds --seed 17 | Out-Null

    for ($f = 0; $f -lt $Folds; $f++) {
        foreach ($b in @("visual","audio","sync")) {
            Usable-Manifest "$run/folds/fold-$f-train.csv" "$run/folds/fold-$f-train-$b.csv" $b
            Usable-Manifest "$run/folds/fold-$f-holdout.csv" "$run/folds/fold-$f-holdout-$b.csv" $b
            # Fewer epochs than the final models: these exist only to score
            # their own holdout, never to be shipped.
            Train-Branch $b "fold$f-$b" "$run/folds/fold-$f-train-$b.csv" `
                "$run/folds/fold-$f-holdout-$b.csv" 4 | Out-Null
        }
        if ((Test-Path "$run/checkpoints/fold$f-visual.pt") -and
            (Test-Path "$run/checkpoints/fold$f-audio.pt") -and
            (Test-Path "$run/checkpoints/fold$f-sync.pt")) {
            Invoke-Ddf "features-oof-fold$f" features export `
                --manifest "$run/folds/fold-$f-holdout-visual.csv" `
                --cache-index "$run/cache-index.csv" --cache-root "$run/cache" `
                --feature-store "$run/features/store.parquet" `
                --report "$run/features/oof-fold$f-report.json" `
                --dataset FakeAVCeleb --run-id "fold$f" --partition-role oof `
                --visual-checkpoint "$run/checkpoints/fold$f-visual.pt" `
                --audio-checkpoint "$run/checkpoints/fold$f-audio.pt" `
                --sync-checkpoint "$run/checkpoints/fold$f-sync.pt" `
                --device cuda | Out-Null
        } else {
            Write-Stage "features-oof-fold$f skipped, a branch checkpoint is missing"
        }
    }

    # ---- 3. fusion ------------------------------------------------------
    if (Test-Path "$run/features/store.parquet") {
        Invoke-Ddf "train-fusion" train fusion `
            --feature-store "$run/features/store.parquet" `
            --output "$run/checkpoints/fusion-logistic.joblib" `
            --metadata "$run/checkpoints/fusion-logistic.json" `
            --model logistic --branches visual audio sync | Out-Null
    }

    # ---- 4. final branch models ----------------------------------------
    foreach ($seed in @(17, 29, 43)) {
        Train-Branch "visual" "final-visual-seed$seed" "$run/split/train-usable.csv" `
            "$run/split/val-usable.csv" 8 $seed | Out-Null
    }
    Train-Branch "audio" "final-audio-seed17" "$run/split/train-audio-usable.csv" `
        "$run/split/val-audio-usable.csv" 6 | Out-Null
    Train-Branch "sync" "final-sync-seed17" "$run/split/train-sync-usable.csv" `
        "$run/split/val-sync-usable.csv" 6 | Out-Null

    # ---- 5. evaluation --------------------------------------------------
    $visual = "$run/checkpoints/final-visual-seed17.pt"
    if (Test-Path $visual) {
        foreach ($t in @(
            @("in-domain-test", "$run/split/test-usable.csv", "FakeAVCeleb", "development_test"),
            @("celebdf", "runs/full-20260904/celebdf-test-manifest.csv", "Celeb-DF-v2", "generalization_celebdf"),
            @("mnw", "runs/full-20260904/mnw-manifest.csv", "MNW", "external_mnw"),
            @("dfdc", "runs/streams-20260905/dfdc-manifest.csv", "DFDC", "generalization_celebdf")
        )) {
            Invoke-Ddf "evaluate-visual-$($t[0])" evaluate branch --branch visual `
                --checkpoint $visual --manifest $t[1] --cache-index "$run/cache-index.csv" `
                --cache-root "$run/cache" --dataset $t[2] --threshold 0.5 `
                --output "$run/evaluation/visual-$($t[0])-metrics.json" `
                --predictions "$run/evaluation/visual-$($t[0])-predictions.csv" `
                --evidence-scope $t[3] --preprocessing-hash $prepHash | Out-Null
        }
    }
    $audio = "$run/checkpoints/final-audio-seed17.pt"
    if (Test-Path $audio) {
        Invoke-Ddf "evaluate-audio-in-domain" evaluate branch --branch audio `
            --checkpoint $audio --manifest "$run/split/test-usable.csv" `
            --cache-index "$run/cache-index.csv" --cache-root "$run/cache" `
            --dataset FakeAVCeleb --threshold 0.5 `
            --output "$run/evaluation/audio-in-domain-test-metrics.json" `
            --predictions "$run/evaluation/audio-in-domain-test-predictions.csv" `
            --evidence-scope development_test --preprocessing-hash $prepHash | Out-Null
    }

    # ---- 6. fusion, the part that has never run on real data ------------
    # In-domain first, then the only cross-corpus set that carries audio.
    # Celeb-DF-v2 and MNW cannot appear here: neither has an audio track, so
    # the audio and sync branches have nothing to read and fusion has no rows.
    Score-Fusion "in-domain-test" "$run/split/test-usable.csv" "FakeAVCeleb" "test"
    Score-Fusion "dfdc" "runs/streams-20260905/dfdc-manifest.csv" "DFDC" "external"

    Write-Stage "program complete"
} catch {
    Write-Stage "program FAILED: $_"
    throw
}
