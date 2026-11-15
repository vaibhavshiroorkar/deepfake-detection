# Export the built deck to PDF, and optionally each slide to PNG, through PowerPoint.
#   powershell -File deliverables/slides/export.ps1                   PDF next to the .pptx
#   powershell -File deliverables/slides/export.ps1 -PngDir some/dir  also one PNG per slide
param([string]$PngDir = "")
$root = Resolve-Path "$PSScriptRoot\..\.."
$src = Join-Path $root "deliverables\build\Deepfake_Detection_Presentation.pptx"
$pp = New-Object -ComObject PowerPoint.Application
$deck = $pp.Presentations.Open($src, $true, $false, $false)
$deck.SaveAs(($src -replace '\.pptx$', '.pdf'), 32)
if ($PngDir) {
    New-Item -ItemType Directory -Force $PngDir | Out-Null
    $i = 0
    foreach ($s in $deck.Slides) { $i++; $s.Export((Join-Path (Resolve-Path $PngDir) "s$i.png"), "PNG", 1280, 720) }
}
$deck.Close(); $pp.Quit()
