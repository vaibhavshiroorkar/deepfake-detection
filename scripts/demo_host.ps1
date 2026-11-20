# Host the exhibition demo from this PC, to be opened on a laptop anywhere through Tailscale.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/demo_host.ps1 -Action start
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/demo_host.ps1 -Action status
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/demo_host.ps1 -Action stop
#
# start: launches the site on 127.0.0.1:8000 with the detector's models loaded first (GPU by default),
# waits until /check answers, then shares it with your own Tailscale devices (`tailscale serve`), if
# Tailscale is installed and logged in. Nothing is opened to the public internet. docs/DEMO.md has
# the one-time setup and what to do on the day.
#
# -Public shares it with the internet instead (`tailscale funnel`), so the static frontend on GitHub Pages
# (scripts/export_site.py) can reach the detector from any browser. Only do this with the hardening in
# docs/SECURITY.md in place; stop turns the funnel off again.
param(
    [ValidateSet("start", "status", "stop")][string]$Action = "status",
    [string]$Device = "cuda",
    [int]$Port = 8000,
    [switch]$Public
)
$Share = if ($Public) { "funnel" } else { "serve" }
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$State = Join-Path $Root "runs\demo-host\state.json"
$Tailscale = "C:\Program Files\Tailscale\tailscale.exe"

function Get-Running {
    if (-not (Test-Path $State)) { return $null }
    $saved = Get-Content $State -Raw | ConvertFrom-Json
    $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$($saved.pid)" -ErrorAction SilentlyContinue
    if ($proc -and $proc.CommandLine -like "*deepfake_detection.webapp*") { return $saved }
    return $null
}

function Test-Page {
    try { return (Invoke-WebRequest -UseBasicParsing -TimeoutSec 5 "http://127.0.0.1:$Port/check").StatusCode -eq 200 }
    catch { return $false }
}

function Show-Share {
    if (-not (Test-Path $Tailscale)) { Write-Output "Tailscale is not installed: open http://127.0.0.1:$Port/check on this PC only."; return }
    & $Tailscale $Share status
}

switch ($Action) {
    "start" {
        if (Get-Running) { Write-Output "Already running."; Show-Share; break }
        $logs = Join-Path $Root "runs\demo-host"
        New-Item -ItemType Directory -Force $logs | Out-Null
        # Keys the server reads (the outside face check and Resend for the contact form)
        # live in .env, which git ignores; they go to the server process only and are never printed.
        $envFile = Join-Path $Root ".env"
        if (Test-Path $envFile) {
            foreach ($line in Get-Content $envFile) {
                if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$') {
                    [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2].Trim().Trim('"'), "Process")
                }
            }
        }
        $env:DDF_SCORER_DEVICE = $Device
        $env:DDF_UPLOAD_LIMIT = "500"        # every request arrives through the local Tailscale proxy
        $python = Join-Path $Root ".venv\Scripts\python.exe"
        $proc = Start-Process -FilePath $python -WorkingDirectory $Root -WindowStyle Hidden -PassThru `
            -ArgumentList "-m deepfake_detection.webapp --host 127.0.0.1 --port $Port --device $Device --preload-detector" `
            -RedirectStandardOutput (Join-Path $logs "server.out.log") -RedirectStandardError (Join-Path $logs "server.err.log")
        @{ pid = $proc.Id; port = $Port; device = $Device; started = (Get-Date).ToString("s") } | ConvertTo-Json |
            Set-Content -Encoding utf8 $State
        Write-Output "Loading the models (about a minute)..."
        $deadline = (Get-Date).AddMinutes(5)
        while (-not (Test-Page)) {
            if ($proc.HasExited) { throw "The server stopped. See runs\demo-host\server.err.log" }
            if ((Get-Date) -gt $deadline) { throw "The site did not answer within 5 minutes. See runs\demo-host\server.err.log" }
            Start-Sleep -Seconds 3
        }
        Write-Output "Ready on this PC: http://127.0.0.1:$Port/check"
        if (Test-Path $Tailscale) { & $Tailscale $Share --bg $Port | Out-Null }
        Show-Share
    }
    "status" {
        $running = Get-Running
        if ($running) { Write-Output "Running (pid $($running.pid), $($running.device)); page answers: $(Test-Page)"; Show-Share }
        else { Write-Output "Not running." }
    }
    "stop" {
        $running = Get-Running
        if (Test-Path $Tailscale) {
            # Whichever was on, public first, then private; turning off one that is already off is not an error.
            foreach ($mode in "funnel", "serve") {
                try { & $Tailscale $mode --https=443 off 2>$null | Out-Null } catch { }
            }
        }
        if ($running) {
            & taskkill /PID $running.pid /T /F | Out-Null       # the server and any ffmpeg it started
            Write-Output "Stopped."
        } else { Write-Output "Not running." }
        if (Test-Path $State) { Remove-Item $State }
    }
}
