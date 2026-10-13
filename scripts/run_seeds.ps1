<#
.SYNOPSIS
Finish the seed replication that keeps every result at `provisional (1 seed)`.

.DESCRIPTION
The frozen protocol asks for three training seeds and the program ran on one, so
every row in docs/research/result-traceability.md reads `provisional (1 seed)`.
That is the single largest gap in the project and it needs no new data.

Some of it was already paid for. `run_program.ps1` trains the visual branch on
seeds 17, 29 and 43, but its stage 5 only ever evaluates seed 17, so two
checkpoints sat on disk unscored. Those are evaluated now and this script skips
them if their metrics exist.

What is left is audio and sync, which ran on seed 17 alone. They are the
expensive part: roughly nine hours for an audio seed and twelve for a sync seed
at full scale, against about forty minutes for a visual one.

Stages, each skipped when its output already exists:

  1. Evaluate the visual seeds on the in-domain test partition and DFDC.
  2. Train audio and sync on the remaining seeds.
  3. Re-export features and refit fusion per seed, then re-run the three
     ablation scripts so the headline numbers carry a seed spread.

Run it overnight. Every stage checks for its own output first, so stopping it
and starting it again loses only the stage that was in flight.

.PARAMETER Seeds
Seeds to complete. The protocol fixes these as 17, 29 and 43.

.PARAMETER Branches
Branches to train. Drop to `visual` for a short run that only fills the
evaluation gap and trains nothing.

.PARAMETER SkipTraining
Evaluate and analyse what is already on disk. Trains nothing.
#>
[CmdletBinding()]
param(
    [int[]]$Seeds = @(17, 29, 43),
    [string[]]$Branches = @("visual", "audio", "sync"),
    [switch]$SkipTraining
)

# Not "Stop": Windows PowerShell turns a native command's stderr into a
# terminating error, and python reports progress there. Exit codes are checked.
$ErrorActionPreference = "Continue"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

$run = "runs/program-20260906"
$python = ".venv/Scripts/python.exe"
$logs = "$run/logs"
New-Item -ItemType Directory -Force -Path $logs | Out-Null

# Read from this run's own cache audit rather than hardcoded. The hash covers
# `code_version`, so a literal copied from another run's audit rejects every
# clip as a preprocessing mismatch. Hardcoding it cost a restart once.
$auditPath = "$run/cache-audit.json"
if (-not (Test-Path $auditPath)) { throw "No cache audit at $auditPath; build the cache first." }
$prepHash = (Get-Content $auditPath -Raw | ConvertFrom-Json).preprocessing_hash
if (-not $prepHash) { throw "Cache audit at $auditPath records no preprocessing hash." }

$transcript = "$logs/seeds.log"

function Write-Stage {
    param([string]$Message)
    $line = "=== $((Get-Date).ToString('s'))  $Message"
    Write-Host $line
    Add-Content -Path $transcript -Value $line -Encoding utf8
}

function Train-Branch {
    param([string]$Branch, [string]$Name, [string]$TrainManifest,
          [string]$ValManifest, [int]$Epochs, [int]$Seed)
    $checkpoint = "$run/checkpoints/$Name.pt"
    if (Test-Path $checkpoint) {
        Write-Stage "train-$Name already present, skipping"
        return
    }
    Write-Stage "train-$Name start"
    $started = Get-Date
    # Sync trains two encoders over temporal tokens, so it needs a smaller batch
    # to stay inside 16 GB of VRAM.
    if ($Branch -eq "sync") { $extra = @("--batch-size","4","--accumulation-steps","4","--heads-epochs","1") }
    else { $extra = @("--batch-size","8","--accumulation-steps","2","--freeze-epochs","1") }
    # Loader workers are a host-RAM question. A visual batch is 8 clips of 16
    # frames at 224x224, which is 77 MB; an audio batch of the same size is 2 MB.
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
    if ($code -ne 0) { Write-Stage "train-$Name FAILED, continuing" }
}

function Evaluate-Visual {
    param([int]$Seed)
    $checkpoint = "$run/checkpoints/final-visual-seed$Seed.pt"
    if (-not (Test-Path $checkpoint)) {
        Write-Stage "evaluate-visual-seed$Seed skipped, no checkpoint"
        return
    }
    # The two partitions the headline rests on. Celeb-DF and MNW are visual-only
    # and already scored on seed 17; add them here if the spread is wanted there.
    foreach ($t in @(
        @("in-domain-test", "$run/split/test-usable.csv", "FakeAVCeleb", "development_test"),
        @("dfdc", "runs/streams-20260905/dfdc-manifest.csv", "DFDC", "generalization_celebdf")
    )) {
        $output = "$run/evaluation/visual-$($t[0])-seed$Seed-metrics.json"
        if (Test-Path $output) {
            Write-Stage "evaluate-visual-$($t[0])-seed$Seed already present, skipping"
            continue
        }
        Write-Stage "evaluate-visual-$($t[0])-seed$Seed start"
        & $python -m deepfake_detection.cli evaluate branch --branch visual `
            --checkpoint $checkpoint --manifest $t[1] `
            --cache-index "$run/cache-index.csv" --cache-root "$run/cache" `
            --dataset $t[2] --threshold 0.5 --output $output `
            --predictions "$run/evaluation/visual-$($t[0])-seed$Seed-predictions.csv" `
            --evidence-scope $t[3] --preprocessing-hash $prepHash `
            --device cuda --batch-size 8 --workers 2 2>&1 |
            Tee-Object -FilePath "$logs/evaluate-visual-$($t[0])-seed$Seed.log"
        Write-Stage "evaluate-visual-$($t[0])-seed$Seed exit=$LASTEXITCODE"
    }
}

function Evaluate-Audio {
    <#
    Score an audio seed on the in-domain test partition.

    A trained checkpoint with no metrics moves no result row, and the first
    version of this script trained the audio and sync seeds without ever scoring
    them. Only in-domain: the program never built an audio cross-corpus
    evaluation, so there is no DFDC audio number at any seed.

    Sync has no equivalent. `ddf evaluate branch` takes visual or audio only, so
    the sync seeds are trained and unscorable without new code.
    #>
    param([int]$Seed)
    $checkpoint = "$run/checkpoints/final-audio-seed$Seed.pt"
    if (-not (Test-Path $checkpoint)) {
        Write-Stage "evaluate-audio-seed$Seed skipped, no checkpoint"
        return
    }
    $output = "$run/evaluation/audio-in-domain-test-seed$Seed-metrics.json"
    if (Test-Path $output) {
        Write-Stage "evaluate-audio-seed$Seed already present, skipping"
        return
    }
    Write-Stage "evaluate-audio-seed$Seed start"
    & $python -m deepfake_detection.cli evaluate branch --branch audio `
        --checkpoint $checkpoint --manifest "$run/split/test-audio-usable.csv" `
        --cache-index "$run/cache-index.csv" --cache-root "$run/cache" `
        --dataset FakeAVCeleb --threshold 0.5 --output $output `
        --predictions "$run/evaluation/audio-in-domain-test-seed$Seed-predictions.csv" `
        --evidence-scope development_test --preprocessing-hash $prepHash `
        --device cuda --batch-size 8 --workers 2 2>&1 |
        Tee-Object -FilePath "$logs/evaluate-audio-seed$Seed.log"
    Write-Stage "evaluate-audio-seed$Seed exit=$LASTEXITCODE"
}

Write-Stage "seeds start seeds=$($Seeds -join ',') branches=$($Branches -join ',')"
try {
    # ---- 1. score the visual checkpoints that already exist -------------
    foreach ($seed in $Seeds) { Evaluate-Visual -Seed $seed }

    # ---- 2. the branches that ran on one seed ---------------------------
    if (-not $SkipTraining) {
        foreach ($seed in $Seeds) {
            if ($Branches -contains "visual") {
                Train-Branch "visual" "final-visual-seed$seed" `
                    "$run/split/train-usable.csv" "$run/split/val-usable.csv" 8 $seed
                Evaluate-Visual -Seed $seed
            }
            if ($Branches -contains "audio") {
                Train-Branch "audio" "final-audio-seed$seed" `
                    "$run/split/train-audio-usable.csv" "$run/split/val-audio-usable.csv" 6 $seed
                Evaluate-Audio -Seed $seed
            }
            if ($Branches -contains "sync") {
                Train-Branch "sync" "final-sync-seed$seed" `
                    "$run/split/train-sync-usable.csv" "$run/split/val-sync-usable.csv" 6 $seed
            }
        }
    }

    # ---- 3. the last pending registry row -------------------------------
    # B-batchnorm-trade is quoted in the paper from a scratch run whose output
    # was never written under runs/, so it stands as `pending`. This is the
    # command that produces it. It trains, but on a few thousand clips rather
    # than the full corpus, so it is the short job here.
    # The path the registry resolves B-batchnorm-trade against. It lives under
    # the Design B run, not this one.
    $batchnorm = "runs/design-b-20260910/evaluation/batchnorm-sweep.json"
    if (-not $SkipTraining -and -not (Test-Path $batchnorm)) {
        Write-Stage "batchnorm-sweep start"
        & $python scripts/batchnorm_sweep.py --run-dir "runs/design-b-20260910" 2>&1 |
            Tee-Object -FilePath "$logs/batchnorm-sweep.log"
        Write-Stage "batchnorm-sweep exit=$LASTEXITCODE"
    }

    # ---- 4. re-run the analysis over whatever now exists ----------------
    # These read the feature stores and cost seconds, so they run every time
    # rather than being skipped: their inputs may have changed above.
    foreach ($script in @(
        "scripts/run_ablation.py",
        "scripts/run_fusion_model_ablation.py",
        "scripts/run_generalization_slices.py",
        "scripts/run_error_analysis.py"
    )) {
        # Corruption re-scores clips through a checkpoint, so it is handled
        # separately below rather than in this read-only analysis loop.
        Write-Stage "analysis $script start"
        & $python $script --run-dir $run 2>&1 |
            Tee-Object -FilePath "$logs/$([System.IO.Path]::GetFileNameWithoutExtension($script)).log"
        Write-Stage "analysis $script exit=$LASTEXITCODE"
    }

    # ---- 5. corruption, which needs the GPU and a checkpoint -------------
    foreach ($corruptionBranch in @("visual", "audio")) {
        Write-Stage "corruption-$corruptionBranch start"
        & $python scripts/run_corruption_eval.py --run-dir $run `
            --branch $corruptionBranch --device cuda 2>&1 |
            Tee-Object -FilePath "$logs/corruption-$corruptionBranch.log"
        Write-Stage "corruption-$corruptionBranch exit=$LASTEXITCODE"
    }

    & $python scripts/update_result_registry.py
    & $python -m deepfake_detection.cli handoff update | Out-Null
    Write-Stage "seeds done"
}
catch {
    Write-Stage "seeds FAILED: $_"
    throw
}
