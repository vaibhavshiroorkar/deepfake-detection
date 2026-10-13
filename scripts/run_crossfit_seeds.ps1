<#
.SYNOPSIS
Cross-fit the folds per seed, which is what 30 of the 36 registry rows need.

.DESCRIPTION
The fusion head is fitted on out-of-fold features, and those come from fold
models that exist at seed 17 only. So every fusion row reads `provisional
(1 seed)` however many seeds the final branches have, and nothing else moves that
count: `run_seeds.ps1` gave the final branches their three seeds, and the head
still sees one.

This is the remaining work: three folds by three branches by two further seeds,
eighteen training runs, plus one out-of-fold export and one head fit per seed.

**Read this before starting it.**

It is long. At measured fold cost, roughly 20 minutes for a visual fold and 8 to
12 for audio and sync, eighteen runs is six to eight hours, and the exports add
about 20 minutes. Every stage skips if its output exists, so stopping and
restarting loses only the stage in flight.

Visual trains with `--workers 0`. A visual batch is 77 MB and prefetching workers
have taken this host out of memory twice. That makes visual folds slower and it
is not optional here.

Do not run anything else heavy alongside it, and do not stack shell wait-loops on
it. Both memory kills in this project's history came from that.

What it buys, stated honestly: the per-seed fusion numbers, and with them the
first `accepted` fusion rows. It will probably also widen the fused seed spread
already measured at 0.1057, because that figure holds the head fixed and varies
only the branches, so it is a floor rather than the full picture.

.PARAMETER Seeds
Seeds to cross-fit. 17 already exists, so the default adds the other two.

.PARAMETER Folds
Fold count. Three is what the existing out-of-fold store used.

.PARAMETER MinFreeCommitGB
Stop cleanly before a stage when Windows has less than this much commit charge
free. A training or export process peaks near 5 GB, and each CUDA process that
exits leaves some commit behind that only a reboot returns, so the free amount
shrinks over a long run. Stopping between stages costs nothing, because a rerun
skips what is done. Being killed mid-stage loses the stage.
#>
[CmdletBinding()]
param(
    [int[]]$Seeds = @(29, 43),
    [int]$Folds = 3,
    [int]$MinFreeCommitGB = 12
)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

$run = "runs/program-20260906"
$python = ".venv/Scripts/python.exe"
$logs = "$run/logs"
New-Item -ItemType Directory -Force -Path $logs | Out-Null

$auditPath = "$run/cache-audit.json"
if (-not (Test-Path $auditPath)) { throw "No cache audit at $auditPath." }
$prepHash = (Get-Content $auditPath -Raw | ConvertFrom-Json).preprocessing_hash
if (-not $prepHash) { throw "Cache audit records no preprocessing hash." }

$transcript = "$logs/crossfit-seeds.log"
function Write-Stage {
    param([string]$Message)
    $line = "=== $((Get-Date).ToString('s'))  $Message"
    Write-Host $line
    Add-Content -Path $transcript -Value $line -Encoding utf8
}

function Assert-Headroom {
    param([string]$Stage)
    $os = Get-CimInstance Win32_OperatingSystem
    $free = $os.FreeVirtualMemory / 1MB
    if ($free -lt $MinFreeCommitGB) {
        $message = "stopping before $Stage, {0:N1} GB commit free against a floor of $MinFreeCommitGB GB. Reboot to return leaked commit, then rerun; finished stages are skipped." -f $free
        Write-Stage $message
        throw $message
    }
}

function Train-Fold {
    param([string]$Branch, [string]$Name, [int]$Fold, [int]$Seed)
    $checkpoint = "$run/checkpoints/$Name.pt"
    if (Test-Path $checkpoint) {
        Write-Stage "$Name already present, skipping"
        return
    }
    # Fold models exist only to score their own holdout, never to ship, so they
    # train for fewer epochs than the final branches.
    if ($Branch -eq "sync") { $extra = @("--batch-size","4","--accumulation-steps","4","--heads-epochs","1") }
    else { $extra = @("--batch-size","8","--accumulation-steps","2","--freeze-epochs","1") }
    # Zero workers throughout. See the note in the synopsis.
    Assert-Headroom $Name
    Write-Stage "$Name start"
    $started = Get-Date
    & $python -m deepfake_detection.cli train $Branch `
        --train-manifest "$run/folds/fold-$Fold-train-$Branch.csv" `
        --validation-manifest "$run/folds/fold-$Fold-holdout-$Branch.csv" `
        --cache-index "$run/cache-index.csv" --cache-root "$run/cache" `
        --dataset FakeAVCeleb --checkpoint $checkpoint `
        --history "$run/checkpoints/$Name-history.json" `
        --run-id $Name --split-hash "program-20260906" --preprocessing-hash $prepHash `
        --device cuda --epochs 4 --learning-rate 0.0001 --weight-decay 0.0001 `
        --patience 2 --workers 0 --seed $Seed @extra 2>&1 |
        Tee-Object -FilePath "$logs/train-$Name.log"
    Write-Stage "$Name exit=$LASTEXITCODE elapsed=$([int]((Get-Date)-$started).TotalSeconds)s"
}

Write-Stage "crossfit start seeds=$($Seeds -join ',') folds=$Folds"
try {
    foreach ($seed in $Seeds) {
        for ($f = 0; $f -lt $Folds; $f++) {
            foreach ($b in @("visual","audio","sync")) {
                Train-Fold $b "fold$f-$b-seed$seed" $f $seed
            }
        }

        # One out-of-fold store per seed, appended fold by fold exactly as
        # run_program.ps1 does, then a head fitted on it. The export now verifies
        # the out-of-fold claim rather than recording it: each fold checkpoint
        # carries the sources it trained on, and a clip whose source is in that
        # set is refused. See fusion/export.py::verify_out_of_fold.
        $store = "$run/features/store-seed$seed.parquet"
        for ($f = 0; $f -lt $Folds; $f++) {
            $ready = $true
            foreach ($b in @("visual","audio","sync")) {
                if (-not (Test-Path "$run/checkpoints/fold$f-$b-seed$seed.pt")) { $ready = $false }
            }
            if (-not $ready) {
                Write-Stage "features-oof-fold$f-seed$seed skipped, a checkpoint is missing"
                continue
            }
            # The report is written only after the store holds this fold's rows,
            # so its presence means the fold is in. Exporting again would
            # recompute every clip and then be refused as duplicate keys.
            if (Test-Path "$run/features/oof-fold$f-seed$seed-report.json") {
                Write-Stage "features-oof-fold$f-seed$seed already present, skipping"
                continue
            }
            Assert-Headroom "features-oof-fold$f-seed$seed"
            Write-Stage "features-oof-fold$f-seed$seed start"
            & $python -m deepfake_detection.cli features export `
                --manifest "$run/folds/fold-$f-holdout-visual.csv" `
                --cache-index "$run/cache-index.csv" --cache-root "$run/cache" `
                --feature-store $store `
                --report "$run/features/oof-fold$f-seed$seed-report.json" `
                --dataset FakeAVCeleb --run-id "fold$f-seed$seed" --partition-role oof `
                --visual-checkpoint "$run/checkpoints/fold$f-visual-seed$seed.pt" `
                --audio-checkpoint "$run/checkpoints/fold$f-audio-seed$seed.pt" `
                --sync-checkpoint "$run/checkpoints/fold$f-sync-seed$seed.pt" `
                --device cuda 2>&1 | Tee-Object -FilePath "$logs/features-oof-fold$f-seed$seed.log"
            Write-Stage "features-oof-fold$f-seed$seed exit=$LASTEXITCODE"
        }

        if (Test-Path $store) {
            Assert-Headroom "train-fusion-seed$seed"
            Write-Stage "train-fusion-seed$seed start"
            & $python -m deepfake_detection.cli train fusion `
                --feature-store $store `
                --output "$run/checkpoints/fusion-logistic-seed$seed.joblib" `
                --metadata "$run/checkpoints/fusion-logistic-seed$seed.json" `
                --model logistic --branches visual audio sync 2>&1 |
                Tee-Object -FilePath "$logs/train-fusion-seed$seed.log"
            Write-Stage "train-fusion-seed$seed exit=$LASTEXITCODE"
        }
    }

    # Re-measure the spread now that the head varies too, not just the branches.
    Assert-Headroom "fusion-seed-spread"
    Write-Stage "fusion-seed-spread start"
    & $python scripts/run_fusion_seed_spread.py --run-dir $run 2>&1 |
        Tee-Object -FilePath "$logs/fusion-seed-spread.log"
    Write-Stage "fusion-seed-spread exit=$LASTEXITCODE"

    & $python scripts/update_result_registry.py
    & $python scripts/update_paper_source.py
    Write-Stage "crossfit done"
}
catch {
    Write-Stage "crossfit FAILED: $_"
    throw
}
