<#
.SYNOPSIS
Fetch the video half of the Microsoft-Northwestern-WITNESS benchmark.

.DESCRIPTION
MNW is the locked external evaluation target. It is evaluation-only by license
and by this project's protocol (docs/data-card.md), and the code enforces that
in src/deepfake_detection/data/guards.py.

The published repository tracks every media file in Git LFS across images, video
and audio. Pulling all of it means tens of thousands of objects, and this
project only detects video. So the checkout is sparse and the LFS pull is
limited to the two video directories that carry usable labels:

  Deepfake_Video/            120 lab clips, 12 generators, fake only, no audio
  AI_media_in_the_wild/Video expert-labelled real-world clips, both classes

The clone is made with LFS smudge disabled so the initial checkout fetches
pointer files only, then the two paths are pulled explicitly. Doing it the other
way round downloads everything before the sparse rules can exclude it.

.PARAMETER Destination
Where to place the checkout. Defaults to data/MNW.

.PARAMETER Commit
Pin to this revision. Recorded alongside the data so an evaluation can be tied
to an exact dataset state; MNW is updated periodically as new generators appear.
#>
[CmdletBinding()]
param(
    [string]$Destination = "data/MNW",
    [string]$Commit = ""
)

# Deliberately not "Stop". Windows PowerShell turns anything a native command
# writes to stderr into a terminating error, and git reports ordinary progress
# there ("Cloning into ...", "Filtering content ..."). Every git call below is
# checked on $LASTEXITCODE instead, which is the only reliable signal.
$ErrorActionPreference = "Continue"

function Invoke-Git {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & git @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "git $($Arguments -join ' ') failed with exit code $LASTEXITCODE"
    }
}

$repository = "https://github.com/microsoft/MNW.git"
$paths = @("Deepfake_Video", "AI_media_in_the_wild/Video")

foreach ($tool in @("git", "git-lfs")) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "$tool is required and was not found on PATH."
    }
}

if (Test-Path $Destination) {
    Write-Host "$Destination already exists; refreshing the tracked paths only."
} else {
    Write-Host "Cloning $repository into $Destination (pointers only)..."
    # GIT_LFS_SKIP_SMUDGE keeps the clone from downloading every media object
    # before sparse-checkout has a chance to narrow the tree.
    $env:GIT_LFS_SKIP_SMUDGE = "1"
    # Deliberately not --filter=blob:none. A blobless partial clone cannot be
    # scanned for LFS pointers ("Could not scan for Git LFS files"), so the
    # later `git lfs pull` finds nothing and silently fetches no media. With
    # smudge skipped the checkout is pointer files only and stays small anyway.
    Invoke-Git clone --no-checkout $repository $Destination
}

Push-Location $Destination
try {
    Invoke-Git sparse-checkout init --cone
    Invoke-Git sparse-checkout set @paths

    $env:GIT_LFS_SKIP_SMUDGE = "1"
    if ($Commit) {
        Invoke-Git checkout $Commit
    } else {
        Invoke-Git checkout
    }

    Remove-Item Env:\GIT_LFS_SKIP_SMUDGE -ErrorAction SilentlyContinue
    foreach ($path in $paths) {
        Write-Host "Pulling LFS objects under $path ..."
        Invoke-Git lfs pull --include="$path/**"
    }

    $pinned = (& git rev-parse HEAD).Trim()
    $videos = Get-ChildItem -Recurse -Include *.mp4, *.mov, *.mkv, *.webm, *.avi |
        Measure-Object
    Write-Host ""
    Write-Host "MNW checkout complete."
    Write-Host "  commit : $pinned"
    Write-Host "  videos : $($videos.Count)"
    Write-Host ""
    Write-Host "Record the commit in docs/data-card.md before evaluating against it."
} finally {
    Pop-Location
    Remove-Item Env:\GIT_LFS_SKIP_SMUDGE -ErrorAction SilentlyContinue
}
