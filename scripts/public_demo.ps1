<#
.SYNOPSIS
Start, inspect, or stop the public demo on this GPU PC.
.DESCRIPTION
Serve a separate runtime containing visual checkpoints and evaluation files.
Dataset directories stay outside that runtime. Cloudflare supplies a temporary URL.
#>
[CmdletBinding()]
param(
    [ValidateSet("start", "status", "stop")]
    [string]$Action = "status",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$workspace = Split-Path -Parent $PSScriptRoot
$hostingRoot = Join-Path $workspace "runs/hosting"
$statePath = Join-Path $hostingRoot "current-demo.json"
$pythonPath = Join-Path $workspace ".venv/Scripts/python.exe"
$tunnelPath = Join-Path $workspace ".venv/Scripts/cloudflared.exe"
$state = if (Test-Path -LiteralPath $statePath) {
    Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
} else { $null }

function Save-DemoState {
    param($Value)
    New-Item -ItemType Directory -Force -Path $hostingRoot | Out-Null
    [System.IO.File]::WriteAllText($statePath, ($Value | ConvertTo-Json -Depth 5), [System.Text.UTF8Encoding]::new($false))
}

function Test-DemoProcess {
    param([int]$ProcessId, [string]$Executable, [string]$Marker)
    if ($ProcessId -le 0) { return $false }
    $process = Get-CimInstance Win32_Process -Filter "ProcessId=$ProcessId"
    if (-not $process) { return $false }
    $command = $process.CommandLine.Replace("/", "\")
    return $command.Contains($Executable.Replace("/", "\")) -and $command.Contains($Marker)
}

function Stop-DemoProcess {
    param([int]$ProcessId, [string]$Executable, [string]$Marker)
    if ($ProcessId -le 0) { return }
    $process = Get-CimInstance Win32_Process -Filter "ProcessId=$ProcessId"
    if (-not $process) { return }
    if (-not (Test-DemoProcess $ProcessId $Executable $Marker)) {
        throw "Process $ProcessId is not this demo's process; it was left running."
    }
    $snapshot = @(Get-CimInstance Win32_Process)
    $owned = [System.Collections.Generic.List[int]]::new()
    $owned.Add($ProcessId)
    for ($index = 0; $index -lt $owned.Count; $index++) {
        foreach ($child in $snapshot) {
            if ($child.ParentProcessId -eq $owned[$index] -and -not $owned.Contains([int]$child.ProcessId)) {
                $owned.Add([int]$child.ProcessId)
            }
        }
    }
    for ($index = $owned.Count - 1; $index -ge 0; $index--) {
        Stop-Process -Id $owned[$index] -ErrorAction SilentlyContinue
    }
}

if ($Action -eq "status") {
    if (-not $state) { Write-Output "No demo has been started."; exit 0 }
    [pscustomobject]@{
        server_running = Test-DemoProcess ([int]$state.server_pid) $pythonPath "deepfake_detection.webapp"
        tunnel_running = Test-DemoProcess ([int]$state.tunnel_pid) $tunnelPath "tunnel"
        url = $state.url
        runtime = $state.runtime_root
        logs = $state.directory
    } | ConvertTo-Json
    exit 0
}

if ($Action -eq "stop") {
    if (-not $state) { Write-Output "No demo has been started."; exit 0 }
    Stop-DemoProcess ([int]$state.tunnel_pid) $tunnelPath "tunnel"
    Stop-DemoProcess ([int]$state.server_pid) $pythonPath "deepfake_detection.webapp"
    $state | Add-Member -NotePropertyName stopped_at -NotePropertyValue (Get-Date -Format o) -Force
    Save-DemoState $state
    Write-Output "Demo stopped."
    exit 0
}

if ($Port -lt 1024 -or $Port -gt 65535) { throw "Port must be between 1024 and 65535." }
if (-not (Test-Path -LiteralPath $pythonPath)) { throw "Install the project environment first. See docs/hosting.md." }
if ($state -and (Test-DemoProcess ([int]$state.server_pid) $pythonPath "deepfake_detection.webapp")) {
    if ($state.url -and (Test-DemoProcess ([int]$state.tunnel_pid) $tunnelPath "tunnel")) {
        Write-Output "Demo is already running: $($state.url)"
        exit 0
    }
    throw "A demo server is already running. Use -Action stop before starting again."
}
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    throw "Port $Port belongs to another process; it was left running."
}

New-Item -ItemType Directory -Force -Path $hostingRoot | Out-Null
$release = "2026.10.3"
$expectedHash = "76b2805253d0d6ffacb092a7acfbee99e7af6917835c0c5410e23f3ef5052da2"
if (-not (Test-Path -LiteralPath $tunnelPath) -or (Get-FileHash -LiteralPath $tunnelPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedHash) {
    $downloadPath = "$tunnelPath.download"
    Invoke-WebRequest -UseBasicParsing -Uri "https://github.com/cloudflare/cloudflared/releases/download/$release/cloudflared-windows-amd64.exe" -OutFile $downloadPath
    if ((Get-FileHash -LiteralPath $downloadPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedHash) {
        throw "Cloudflared download digest mismatch. The downloaded file was not executed."
    }
    Move-Item -LiteralPath $downloadPath -Destination $tunnelPath -Force
}

$directory = Join-Path $hostingRoot ("public-demo-" + (Get-Date -Format "yyyyMMddTHHmmss"))
$runtimeRoot = Join-Path $directory "runtime"
New-Item -ItemType Directory -Force -Path $runtimeRoot | Out-Null
foreach ($pattern in @("final-visual-seed*.pt", "final-visual-convnext-seed*.pt", "final-visual*-history.json")) {
    $destination = Join-Path $runtimeRoot "runs/program-20260906/checkpoints"
    New-Item -ItemType Directory -Force -Path $destination | Out-Null
    Get-ChildItem -LiteralPath (Join-Path $workspace "runs/program-20260906/checkpoints") -Filter $pattern -File |
        Copy-Item -Destination $destination
}
$evaluationRoot = Join-Path $runtimeRoot "runs/program-20260906/evaluation"
New-Item -ItemType Directory -Force -Path $evaluationRoot | Out-Null
Get-ChildItem -LiteralPath (Join-Path $workspace "runs/program-20260906/evaluation") -Filter "visual*" -File |
    Where-Object { $_.Extension -in @(".json", ".csv") } | Copy-Item -Destination $evaluationRoot
$densityRoot = Join-Path $runtimeRoot "runs/probe-20260914"
New-Item -ItemType Directory -Force -Path $densityRoot | Out-Null
Copy-Item -LiteralPath (Join-Path $workspace "runs/probe-20260914/authenticity-density.npz") -Destination $densityRoot
$env:DDF_PROJECT_ROOT = $runtimeRoot
Remove-Item Env:DDF_ADMIN_TOKEN -ErrorAction SilentlyContinue
& $pythonPath -c "from deepfake_detection.webapp.settings import Settings,save; save(Settings(default_model='ensemble', enabled_models=('ensemble','visual-best'), max_upload_mb=90))"
if ($LASTEXITCODE -ne 0) { throw "Demo configuration failed." }

$state = [pscustomobject]@{
    port = $Port; host = "127.0.0.1"; server_pid = 0; tunnel_pid = 0; url = $null
    directory = $directory; runtime_root = $runtimeRoot; started_at = (Get-Date -Format o)
}
try {
    $server = Start-Process -FilePath $pythonPath -ArgumentList @("-m", "deepfake_detection.webapp", "--host", "127.0.0.1", "--port", "$Port", "--device", "cuda", "--preload") -WorkingDirectory $runtimeRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $directory "server.stdout.log") -RedirectStandardError (Join-Path $directory "server.stderr.log") -PassThru
    $state.server_pid = $server.Id
    Save-DemoState $state
    $ready = $false
    for ($attempt = 0; $attempt -lt 180; $attempt++) {
        if (-not (Test-DemoProcess $server.Id $pythonPath "deepfake_detection.webapp")) { throw "Server exited. Inspect $directory/server.stderr.log." }
        try {
            $config = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/config" -TimeoutSec 3
            if ($config.samples.Count -ne 0 -or $config.admin_enabled) { throw "The public runtime must not expose local samples or admin access." }
            $ready = $true
            break
        } catch [System.Net.WebException] { Start-Sleep -Seconds 2 }
    }
    if (-not $ready) { throw "Server startup timed out. Inspect $directory/server.stderr.log." }
    $tunnel = Start-Process -FilePath $tunnelPath -ArgumentList @("tunnel", "--url", "http://127.0.0.1:$Port", "--no-autoupdate", "--protocol", "http2") -WorkingDirectory $runtimeRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $directory "tunnel.stdout.log") -RedirectStandardError (Join-Path $directory "tunnel.stderr.log") -PassThru
    $state.tunnel_pid = $tunnel.Id
    Save-DemoState $state
    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        if (-not (Test-DemoProcess $tunnel.Id $tunnelPath "tunnel")) { throw "Tunnel exited. Inspect $directory/tunnel.stderr.log." }
        $log = Get-Content -LiteralPath (Join-Path $directory "tunnel.stderr.log") -Raw -ErrorAction SilentlyContinue
        if ($log -match 'https://[a-z0-9-]+\.trycloudflare\.com') { $state.url = $Matches[0]; break }
        Start-Sleep -Seconds 2
    }
    if (-not $state.url) { throw "Tunnel URL was not issued. Inspect $directory/tunnel.stderr.log." }
    Save-DemoState $state
    Write-Output "Public demo: $($state.url)"
    Write-Output "Keep this PC awake. Stop it with scripts/public_demo.ps1 -Action stop."
} catch {
    Stop-DemoProcess ([int]$state.tunnel_pid) $tunnelPath "tunnel"
    Stop-DemoProcess ([int]$state.server_pid) $pythonPath "deepfake_detection.webapp"
    throw
}
