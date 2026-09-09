<#
.SYNOPSIS
Run one tier of the training program end to end, unattended.

.DESCRIPTION
Sequential on purpose. Every stage wants the whole GPU, and two at once only
contend for its memory. The cache build is the exception: it is shard-parallel
because it is CPU-bound on video decode, so several workers do help there.

The script is safe to re-run. `ddf cache build --skip-cached` reuses everything
already on disk, training overwrites its own checkpoint, and the evaluations are
pure functions of a checkpoint and a manifest. An interrupted run is resumed by
starting it again, not by unpicking where it stopped.

`ddf handoff update` runs after every stage, so docs/handoff.md describes the
state that exists rather than the state someone last remembered to write down.

.PARAMETER Tier
"pilot" (a stratified 4,000-clip subsample, ~1 day) or "full" (all 21,544).

.PARAMETER Shards
How many concurrent cache workers. Three saturates the CPU on this host.

.PARAMETER SkipCache
Assume the cache is already built and go straight to training.
#>
[CmdletBinding()]
param(
    [ValidateSet("pilot", "full")][string]$Tier = "pilot",
    [int]$Shards = 3,
    [switch]$SkipCache,
    [string]$PreprocessingHash = "ac79f71e7614c96610e897eb011e01129b193da4482f415d037c6cc8c17638ec"
)

# Not "Stop": Windows PowerShell turns a native command's stderr into a
# terminating error, and both python and git report progress there. Stages are
# checked on their exit code instead.
$ErrorActionPreference = "Continue"

$ErrorView = "NormalView"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $here
Set-Location $root

$run = "runs/full-20260904"
$python = ".venv/Scripts/python.exe"
$logDirectory = "$run/logs"
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
New-Item -ItemType Directory -Force -Path "$run/checkpoints" | Out-Null
New-Item -ItemType Directory -Force -Path "$run/evaluation" | Out-Null

$transcript = "$logDirectory/supervise-$Tier.log"
$splitDirectory = if ($Tier -eq "pilot") { "$run/pilot-split" } else { "$run/split" }

function Write-Stage {
    param([string]$Message)
    $stamp = (Get-Date).ToString("s")
    $line = "=== $stamp  $Message"
    Write-Host $line
    Add-Content -Path $transcript -Value $line -Encoding utf8
}

function Invoke-Stage {
    <#
    Run one ddf command. Exit code 2 means partial output (some clips failed,
    some rows lacked evidence) and is expected on real data, so it is recorded
    and the program continues. Anything else stops the tier, because a training
    stage that failed makes every stage after it meaningless.
    #>
    param(
        [string]$Name,
        [Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments
    )
    Write-Stage "$Name start"
    $started = Get-Date
    & $python -m deepfake_detection.cli @Arguments 2>&1 |
        Tee-Object -FilePath "$logDirectory/$Name.log"
    $code = $LASTEXITCODE
    $elapsed = [int]((Get-Date) - $started).TotalSeconds
    Write-Stage "$Name exit=$code elapsed=${elapsed}s"
    if ($code -gt 2) {
        throw "$Name failed with exit code $code"
    }
    & $python -m deepfake_detection.cli handoff update | Out-Null
    return $code
}

function Build-Cache {
    param([string]$Name, [string]$Manifest, [string]$DatasetName)

    Write-Stage "cache-$Name start ($Shards shards)"
    $started = Get-Date
    $jobs = @()
    for ($index = 0; $index -lt $Shards; $index++) {
        $jobs += Start-Process -FilePath $python -PassThru -NoNewWindow `
            -RedirectStandardOutput "$logDirectory/cache-$Name-shard$index.log" `
            -RedirectStandardError "$logDirectory/cache-$Name-shard$index.err" `
            -ArgumentList @(
                "-m", "deepfake_detection.cli", "cache", "build",
                "--manifest", $Manifest,
                "--dataset-root", "data",
                "--cache-root", "$run/cache",
                "--index", "$run/cache-index-$Name-shard$index.csv",
                "--audit", "$run/cache-audit-$Name-shard$index.json",
                "--dataset", $DatasetName,
                "--device", "cuda",
                "--code-version", "full-v1",
                "--skip-cached",
                "--shard", "$index/$Shards"
            )
    }
    $jobs | Wait-Process
    $elapsed = [int]((Get-Date) - $started).TotalSeconds
    Write-Stage "cache-$Name shards done elapsed=${elapsed}s"

    $indexes = (0..($Shards - 1) | ForEach-Object { "$run/cache-index-$Name-shard$_.csv" })
    $audits = (0..($Shards - 1) | ForEach-Object { "$run/cache-audit-$Name-shard$_.json" })
    Invoke-Stage "cache-merge-$Name" cache merge `
        --indexes @indexes --audits @audits `
        --index "$run/cache-index-$Name.csv" --audit "$run/cache-audit-$Name.json" | Out-Null
}

function Invoke-Training {
    param([string]$ConfigName)
    Write-Stage "train-$ConfigName start"
    $started = Get-Date
    & $python -m deepfake_detection.cli run --config "$run/configs/$ConfigName.yaml" 2>&1 |
        Tee-Object -FilePath "$logDirectory/train-$ConfigName.log"
    $code = $LASTEXITCODE
    $elapsed = [int]((Get-Date) - $started).TotalSeconds
    Write-Stage "train-$ConfigName exit=$code elapsed=${elapsed}s"
    & $python -m deepfake_detection.cli handoff update | Out-Null
    if ($code -ne 0) {
        # A failed branch is recorded and skipped rather than stopping the tier.
        # Its checkpoint simply will not exist, and the evaluations that need it
        # are skipped in turn, which is more useful than losing the runs that
        # did succeed.
        Write-Stage "train-$ConfigName FAILED, continuing with the remaining runs"
        return $false
    }
    return $true
}

function Invoke-Evaluation {
    param(
        [string]$Name,
        [string]$Checkpoint,
        [string]$Manifest,
        [string]$CacheIndex,
        [string]$DatasetName,
        [string]$Scope,
        [string]$Branch = "visual"
    )
    if (-not (Test-Path $Checkpoint)) {
        Write-Stage "evaluate-$Name skipped, no checkpoint at $Checkpoint"
        return
    }
    Invoke-Stage "evaluate-$Name" evaluate branch `
        --branch $Branch `
        --checkpoint $Checkpoint `
        --manifest $Manifest `
        --cache-index $CacheIndex `
        --cache-root "$run/cache" `
        --dataset $DatasetName `
        --threshold 0.5 `
        --output "$run/evaluation/$Name-metrics.json" `
        --predictions "$run/evaluation/$Name-predictions.csv" `
        --evidence-scope $Scope | Out-Null
}

Write-Stage "supervisor start tier=$Tier shards=$Shards"

try {
    if (-not $SkipCache) {
        $trainingManifest = if ($Tier -eq "pilot") {
            "$run/pilot-cache-manifest.csv"
        } else {
            "$run/fakeavceleb-all.csv"
        }
        Build-Cache -Name $Tier -Manifest $trainingManifest -DatasetName "FakeAVCeleb"
        Build-Cache -Name "celebdf" -Manifest "$run/celebdf-test-manifest.csv" -DatasetName "Celeb-DF-v2"
        Build-Cache -Name "mnw" -Manifest "$run/mnw-manifest.csv" -DatasetName "MNW"
    }

    # Training reads one index, so the per-dataset indexes are folded together.
    Invoke-Stage "cache-merge-all" cache merge `
        --indexes "$run/cache-index-$Tier.csv" "$run/cache-index-celebdf.csv" "$run/cache-index-mnw.csv" `
        --audits "$run/cache-audit-$Tier.json" "$run/cache-audit-celebdf.json" "$run/cache-audit-mnw.json" `
        --index "$run/cache-index.csv" --audit "$run/cache-audit.json" | Out-Null

    # Visual first, then audio, then sync. Alphabetical order would run the two
    # expensive branches before the cheap one, so an interruption partway would
    # leave the cheapest and most useful result unbuilt. Within a branch the
    # seeds run in order.
    $branchOrder = @("visual", "audio", "sync")
    # Filter each partition to clips the branch can actually read. A clip whose
    # primary face track was unstable has no visual view, and the loader raises
    # on it mid-epoch. The audit records every dropped clip and its reason, so
    # the abstention rate stays reportable.
    foreach ($partition in @("train", "val", "test")) {
        Invoke-Stage "usable-visual-$partition" manifest usable `
            --manifest "$splitDirectory/$partition.csv" `
            --cache-index "$run/cache-index.csv" `
            --cache-root "$run/cache" `
            --output "$splitDirectory/$partition-usable.csv" `
            --audit "$run/usable-visual-$partition-audit.json" `
            --dataset "FakeAVCeleb" --branch visual `
            --preprocessing-hash $PreprocessingHash | Out-Null
    }
    foreach ($branch in @("audio", "sync")) {
        foreach ($partition in @("train", "val")) {
            Invoke-Stage "usable-$branch-$partition" manifest usable `
                --manifest "$splitDirectory/$partition.csv" `
                --cache-index "$run/cache-index.csv" `
                --cache-root "$run/cache" `
                --output "$splitDirectory/$partition-$branch-usable.csv" `
                --audit "$run/usable-$branch-$partition-audit.json" `
                --dataset "FakeAVCeleb" --branch $branch `
                --preprocessing-hash $PreprocessingHash | Out-Null
        }
    }

    $configs = Get-ChildItem "$run/configs/$Tier-*.yaml" |
        Sort-Object @{ Expression = {
            $index = $branchOrder.IndexOf(($_.BaseName -split "-")[1])
            if ($index -lt 0) { $branchOrder.Count } else { $index }
        } }, Name
    foreach ($config in $configs) {
        Invoke-Training $config.BaseName | Out-Null
    }

    # Evaluations. Visual only against Celeb-DF and MNW: MNW has no audio track
    # at all, so the audio and sync branches have nothing to read there.
    $visual = "$run/checkpoints/$Tier-visual-efficientnet-b0-seed17.pt"
    Invoke-Evaluation -Name "$Tier-visual-validation" -Checkpoint $visual `
        -Manifest "$splitDirectory/val-usable.csv" -CacheIndex "$run/cache-index.csv" `
        -DatasetName "FakeAVCeleb" -Scope "development_validation"
    Invoke-Evaluation -Name "$Tier-visual-test" -Checkpoint $visual `
        -Manifest "$splitDirectory/test-usable.csv" -CacheIndex "$run/cache-index.csv" `
        -DatasetName "FakeAVCeleb" -Scope "development_test"
    Invoke-Evaluation -Name "$Tier-visual-celebdf" -Checkpoint $visual `
        -Manifest "$run/celebdf-test-manifest.csv" -CacheIndex "$run/cache-index.csv" `
        -DatasetName "Celeb-DF-v2" -Scope "generalization_celebdf"
    Invoke-Evaluation -Name "$Tier-visual-mnw" -Checkpoint $visual `
        -Manifest "$run/mnw-manifest.csv" -CacheIndex "$run/cache-index.csv" `
        -DatasetName "MNW" -Scope "external_mnw"

    $audio = "$run/checkpoints/$Tier-audio-wav2vec2-base-seed17.pt"
    Invoke-Evaluation -Name "$Tier-audio-validation" -Checkpoint $audio -Branch "audio" `
        -Manifest "$splitDirectory/val-usable.csv" -CacheIndex "$run/cache-index.csv" `
        -DatasetName "FakeAVCeleb" -Scope "development_validation"

    Write-Stage "supervisor complete tier=$Tier"
} catch {
    Write-Stage "supervisor FAILED: $_"
    throw
}
