<#
.SYNOPSIS
Live progress for a supervised training run, in its own console window.

.DESCRIPTION
The CLI stages print nothing while they work, so progress has to be derived
from what they leave behind:

  - Cache stages: the count of .npz files under the cache root against the
    number of rows in the manifests being built.
  - Training stages: MLflow records one metric row per epoch as it finishes, so
    the highest step for a run's training.loss is the epoch it is on.
  - Stage and timing: the supervisor's own transcript.

Everything is read-only. Closing this window does not touch the run.

.PARAMETER Tier
Which tier's transcript to follow. Defaults to full.

.PARAMETER IntervalSeconds
Refresh period. The cache count is a directory walk over ~20k files, so a very
short interval costs more than it shows.
#>
[CmdletBinding()]
param(
    [ValidateSet("pilot", "full")][string]$Tier = "full",
    [int]$IntervalSeconds = 20
)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

$run = "runs/full-20260904"
$cacheRoot = "$run/cache"
$transcript = "$run/logs/supervise-$Tier.log"
$database = "mlflow.db"

# Total clips the full program caches: FakeAVCeleb plus the two external sets.
$expectedClips = @{ pilot = 4649; full = 22193 }[$Tier]

function Get-Bar {
    param([double]$Fraction, [int]$Width = 44)
    if ($Fraction -lt 0) { $Fraction = 0 }
    if ($Fraction -gt 1) { $Fraction = 1 }
    $filled = [int][Math]::Round($Fraction * $Width)
    return "[" + ("#" * $filled) + ("-" * ($Width - $filled)) + "]"
}

function Get-Duration {
    param([TimeSpan]$Span)
    if ($Span.TotalHours -ge 1) {
        return "{0:0}h {1:00}m" -f [Math]::Floor($Span.TotalHours), $Span.Minutes
    }
    return "{0:0}m {1:00}s" -f [Math]::Floor($Span.TotalMinutes), $Span.Seconds
}

function Get-Stages {
    if (-not (Test-Path $transcript)) { return @() }
    return Get-Content $transcript -ErrorAction SilentlyContinue |
        Where-Object { $_ -match "^﻿?=== (\S+)\s+(.+)$" } |
        ForEach-Object {
            if ($_ -match "^﻿?=== (\S+)\s+(.+)$") {
                [pscustomobject]@{ Time = [datetime]::Parse($Matches[1]); Text = $Matches[2] }
            }
        }
}

function Get-EpochProgress {
    <#
    The epoch a training run has finished, read straight from MLflow. Returns
    $null when the run has not logged anything yet, which is normal for the
    first few minutes while the backbone loads and the first epoch runs.
    #>
    param([string]$RunName)
    if (-not (Test-Path $database)) { return $null }
    $query = @"
SELECT MAX(m.step), COUNT(*) FROM metrics m
JOIN runs r ON r.run_uuid = m.run_uuid
WHERE r.name = '$RunName' AND m.key = 'training.loss'
"@
    try {
        $result = & ".venv/Scripts/python.exe" -c @"
import sqlite3, sys
c = sqlite3.connect('file:$database?mode=ro', uri=True)
row = c.execute('''SELECT MAX(m.step) FROM metrics m JOIN runs r ON r.run_uuid=m.run_uuid
                   WHERE r.name=? AND m.key='training.loss' ''', ('$RunName',)).fetchone()
loss = c.execute('''SELECT m.value FROM metrics m JOIN runs r ON r.run_uuid=m.run_uuid
                    WHERE r.name=? AND m.key='validation.loss' ORDER BY m.step DESC LIMIT 1''',
                 ('$RunName',)).fetchone()
print((row[0] if row and row[0] is not None else -1), (loss[0] if loss else ''))
"@ 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $result) { return $null }
        $parts = "$result".Trim() -split "\s+"
        return [pscustomobject]@{
            Epoch = [int]$parts[0]
            ValidationLoss = if ($parts.Count -gt 1) { $parts[1] } else { "" }
        }
    } catch { return $null }
}

function Get-ConfiguredEpochs {
    param([string]$ConfigName)
    $path = "$run/configs/$ConfigName.yaml"
    if (-not (Test-Path $path)) { return 0 }
    $line = Select-String -Path $path -Pattern "^\s+epochs:\s*(\d+)" |
        Select-Object -First 1
    if ($line) { return [int]$line.Matches[0].Groups[1].Value }
    return 0
}

# Listed in the order the supervisor actually runs them (visual, then audio,
# then sync), not alphabetically, so the queue reads as a schedule.
$branchOrder = @("visual", "audio", "sync")
$allConfigs = Get-ChildItem "$run/configs/$Tier-*.yaml" -ErrorAction SilentlyContinue |
    Sort-Object @{ Expression = {
        $index = $branchOrder.IndexOf(($_.BaseName -split "-")[1])
        if ($index -lt 0) { $branchOrder.Count } else { $index }
    } }, Name |
    Select-Object -ExpandProperty BaseName
$previousCount = $null
$previousTime = $null
$rateHistory = New-Object System.Collections.Generic.Queue[double]

$host.UI.RawUI.WindowTitle = "deepfake-generalization: $Tier run"

while ($true) {
    $stages = Get-Stages
    $last = $stages | Select-Object -Last 1
    $started = ($stages | Select-Object -First 1)
    $now = Get-Date

    $cached = 0
    if (Test-Path $cacheRoot) {
        $cached = (Get-ChildItem $cacheRoot -Recurse -Filter *.npz -ErrorAction SilentlyContinue |
            Measure-Object).Count
    }

    # Smoothed rate, because a directory walk lands unevenly.
    if ($null -ne $previousCount -and $cached -gt $previousCount) {
        $seconds = ($now - $previousTime).TotalSeconds
        if ($seconds -gt 0) {
            $rateHistory.Enqueue(($cached - $previousCount) / $seconds * 60)
            while ($rateHistory.Count -gt 10) { [void]$rateHistory.Dequeue() }
        }
    }
    if ($null -eq $previousCount -or $cached -ne $previousCount) {
        $previousCount = $cached
        $previousTime = $now
    }
    $rate = if ($rateHistory.Count) {
        ($rateHistory | Measure-Object -Average).Average
    } else { 0 }

    Clear-Host
    Write-Host ""
    Write-Host "  deepfake-generalization  --  $Tier tier" -ForegroundColor Cyan
    if ($started) {
        Write-Host "  running for $(Get-Duration ($now - $started.Time))  |  $(Get-Date -Format 'HH:mm:ss')"
    }
    Write-Host ""

    # ---- cache ----
    $fraction = if ($expectedClips) { $cached / $expectedClips } else { 0 }
    Write-Host "  PREPROCESSING CACHE" -ForegroundColor Yellow
    Write-Host ("  {0} {1,6:P1}" -f (Get-Bar $fraction), $fraction)
    $remaining = [Math]::Max(0, $expectedClips - $cached)
    $eta = if ($rate -gt 0.5 -and $remaining -gt 0) {
        Get-Duration ([TimeSpan]::FromMinutes($remaining / $rate))
    } elseif ($remaining -eq 0) { "done" } else { "..." }
    Write-Host ("  {0:N0} / {1:N0} clips   {2:N0}/min   eta {3}" -f $cached, $expectedClips, $rate, $eta)
    Write-Host ""

    # ---- training ----
    Write-Host "  TRAINING RUNS" -ForegroundColor Yellow
    $completed = @($stages | Where-Object { $_.Text -match "^train-(\S+) exit=0" } |
        ForEach-Object { if ($_.Text -match "^train-(\S+) exit=") { $Matches[1] } })
    $activeConfig = $null
    if ($last -and $last.Text -match "^train-(\S+) start$") { $activeConfig = $Matches[1] }

    foreach ($config in $allConfigs) {
        $short = $config -replace "^$Tier-", ""
        if ($completed -contains $config) {
            Write-Host ("    done     {0}" -f $short) -ForegroundColor Green
        } elseif ($config -eq $activeConfig) {
            $total = Get-ConfiguredEpochs $config
            $progress = Get-EpochProgress $config
            $elapsed = Get-Duration ($now - $last.Time)
            if ($progress -and $progress.Epoch -ge 0 -and $total -gt 0) {
                $epochFraction = [Math]::Min(1, ($progress.Epoch + 1) / $total)
                Write-Host ("    RUNNING  {0}" -f $short) -ForegroundColor Cyan
                Write-Host ("             {0} epoch {1}/{2}  {3}  val {4}" -f `
                    (Get-Bar $epochFraction 28), ($progress.Epoch + 1), $total, $elapsed, $progress.ValidationLoss)
            } else {
                Write-Host ("    RUNNING  {0}  (starting, {1})" -f $short, $elapsed) -ForegroundColor Cyan
            }
        } else {
            Write-Host ("    queued   {0}" -f $short) -ForegroundColor DarkGray
        }
    }
    Write-Host ""

    # ---- current stage ----
    Write-Host "  CURRENT STAGE" -ForegroundColor Yellow
    if ($last) {
        Write-Host ("    {0}  ({1} ago)" -f $last.Text, (Get-Duration ($now - $last.Time)))
    } else {
        Write-Host "    waiting for the supervisor to write its transcript"
    }

    $failures = @($stages | Where-Object { $_.Text -match "FAILED" })
    if ($failures.Count) {
        Write-Host ""
        Write-Host "  FAILURES" -ForegroundColor Red
        $failures | Select-Object -Last 4 | ForEach-Object {
            Write-Host ("    {0}" -f $_.Text) -ForegroundColor Red
        }
    }

    $free = (Get-PSDrive -Name ($root.Substring(0, 1)) -ErrorAction SilentlyContinue).Free
    Write-Host ""
    if ($free) {
        Write-Host ("  disk free {0:N0} GB   refresh {1}s   Ctrl+C to close (the run keeps going)" -f `
            ($free / 1GB), $IntervalSeconds) -ForegroundColor DarkGray
    }

    if ($last -and $last.Text -match "supervisor complete") {
        Write-Host ""
        Write-Host "  RUN COMPLETE" -ForegroundColor Green
        break
    }
    Start-Sleep -Seconds $IntervalSeconds
}
