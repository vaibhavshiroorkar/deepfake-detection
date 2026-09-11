<#
Design B training: the three visual streams, then the two audiovisual streams.

Sequential, not parallel. One RTX 5070 Ti with 16 GB, and Xception at 224 pixels
over 16 frames already needs a batch of 4, so two runs sharing the card would
mean cutting both batch sizes and gaining nothing.

Loader workers matter more here than anywhere else in the pipeline. A visual
batch is 16 frames of 3x224x224 float32, about 77 MB, so the main thread spends
its time decompressing npz files: measured at 65 ms per clip, an epoch over
7,637 clips is 8 minutes of pure I/O and the GPU sat at 15 percent utilisation
waiting for it. Workers fix that and can also die on batches this size, which is
why each stream is attempted with workers and then retried single-process,
matching run_program.ps1.

    pwsh -File scripts/train_design_b.ps1 -RunDir runs/design-b-20260910
#>
param(
    [string]$RunDir = "runs/design-b-20260910",
    [string]$SourceRun = "runs/program-20260906",
    [int]$Epochs = 10,
    [int]$Workers = 2,
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Continue"
$python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"

# Read the hash from the run that built the cache. Hardcoding it once cost a
# full program run: a value copied from an earlier code_version rejected every
# clip, because the hash covers the code version as well as the settings.
$audit = Join-Path $SourceRun "cache-audit.json"
if (-not (Test-Path $audit)) { throw "No cache audit at $audit" }
$prepHash = (Get-Content $audit -Raw | ConvertFrom-Json).preprocessing_hash
if (-not $prepHash) { throw "$audit has no preprocessing_hash" }
Write-Host "preprocessing hash: $prepHash"

$checkpoints = Join-Path $RunDir "checkpoints"
$logs = Join-Path $RunDir "logs"
foreach ($d in @($checkpoints, $logs)) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Force $d | Out-Null }
}

# Each stream trains on the manifest of clips it can actually read. These are
# not interchangeable: `train-usable.csv` is the visual-usable set at 7,637
# clips while `train-sync-usable.csv` is 7,628, and the nine-clip difference is
# enough to kill a lip-sync run mid-epoch on a clip that has no mouth view.
#
# Emotion reads the face and audio views, and every visual-usable clip here is
# also audio-usable, so the visual manifest is already the intersection.
function Get-Manifests {
    param([string]$Stream)
    $suffix = if ($Stream -like "*lipsync*") { "sync-usable" } else { "usable" }
    return @(
        (Join-Path $SourceRun "split	rain-$suffix.csv"),
        (Join-Path $SourceRun "splital-$suffix.csv")
    )
}

function Invoke-Stream {
    param([string]$Name, [string[]]$Extra)

    $checkpoint = Join-Path $checkpoints "$Name.pt"
    if (Test-Path $checkpoint) {
        Write-Host "$Name : already trained, skipping"
        return
    }
    $log = Join-Path $logs "$Name.log"
    Write-Host "=== $Name  ($(Get-Date -Format HH:mm:ss)) ==="

    # Workers first, then single-process. A dead worker takes the run down with
    # a DataLoader error rather than a wrong answer, so retrying costs one
    # wasted epoch and never a silently degraded model.
    $manifests = Get-Manifests -Stream $Name
    foreach ($attempt in @($Workers, 0)) {
        $common = @(
            "--train-manifest", $manifests[0], "--validation-manifest", $manifests[1],
            "--cache-index", (Join-Path $SourceRun "cache-index.csv"),
            "--cache-root", (Join-Path $SourceRun "cache"),
            "--dataset", "FakeAVCeleb",
            "--checkpoint", $checkpoint,
            "--history", (Join-Path $checkpoints "$Name-history.json"),
            "--run-id", $Name,
            "--split-hash", (Split-Path $RunDir -Leaf),
            "--preprocessing-hash", $prepHash,
            "--device", $Device, "--epochs", $Epochs, "--workers", $attempt
        )
        & $python -m deepfake_detection.cli @Extra @common 2>&1 | Tee-Object -FilePath $log
        if ($LASTEXITCODE -eq 0) {
            Write-Host "$Name done ($(Get-Date -Format HH:mm:ss))"
            return
        }
        if ($attempt -eq 0) {
            Write-Host "$Name FAILED with exit $LASTEXITCODE"
            return
        }
        Write-Host "$Name failed with $attempt workers, retrying single-process"
    }
}

# DINOv3 stays frozen for the whole run. That is the reason it is here: a
# fine-tuned backbone can absorb the corpus it trains on, which is how the
# EfficientNet baseline reached 0.9742 in-domain and then called 130 of 155
# genuine Celeb-DF videos fake. Only the head is fitted, so the features stay
# the self-supervised ones.
Invoke-Stream -Name "visual-dinov3" -Extra @(
    "train", "visual-stream", "--backbone", "dinov3", "--freeze-backbone",
    "--batch-size", "8", "--accumulation-steps", "2", "--frame-chunk-size", "8",
    "--learning-rate", "1e-3")

Invoke-Stream -Name "visual-efficientnet" -Extra @(
    "train", "visual-stream", "--backbone", "efficientnet", "--freeze-epochs", "2",
    "--batch-size", "8", "--accumulation-steps", "2", "--frame-chunk-size", "8")

Invoke-Stream -Name "visual-xception" -Extra @(
    "train", "visual-stream", "--backbone", "xception", "--freeze-epochs", "2",
    "--batch-size", "4", "--accumulation-steps", "4", "--frame-chunk-size", "4")

# The audiovisual pair. Wav2Vec2 stands in for AV-HuBERT on the audio side,
# which is the substitution to revisit first: it encodes phonetic content rather
# than audiovisual correspondence, and every cross-modal stream trained here so
# far has sat at chance.
Invoke-Stream -Name "stream-lipsync" -Extra @(
    "train", "stream", "--stream", "lipsync", "--freeze-epochs", "2",
    "--batch-size", "8", "--accumulation-steps", "2")

Invoke-Stream -Name "stream-emotion" -Extra @(
    "train", "stream", "--stream", "emotion", "--freeze-epochs", "2",
    "--batch-size", "8", "--accumulation-steps", "2")

Write-Host "=== all streams done ($(Get-Date -Format HH:mm:ss)) ==="
