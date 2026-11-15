# Export a built deck or the poster to PDF, and optionally each slide to PNG, through PowerPoint.
#   powershell -File deliverables/slides/export.ps1                                   the deck
#   powershell -File deliverables/slides/export.ps1 -Name Deepfake_Detection_Poster   the poster
#   ... -PngDir some/dir                                                               also one PNG per slide
param([string]$Name = "Deepfake_Detection_Presentation", [string]$PngDir = "", [int]$Width = 1280)
$root = Resolve-Path "$PSScriptRoot\..\.."
$src = Join-Path $root "deliverables\build\$Name.pptx"
$pp = New-Object -ComObject PowerPoint.Application
$deck = $pp.Presentations.Open($src, $true, $false, $false)
$deck.SaveAs(($src -replace '\.pptx$', '.pdf'), 32)
if ($PngDir) {
    New-Item -ItemType Directory -Force $PngDir | Out-Null
    $h = [int]($Width * $deck.PageSetup.SlideHeight / $deck.PageSetup.SlideWidth)
    $i = 0
    foreach ($s in $deck.Slides) { $i++; $s.Export((Join-Path (Resolve-Path $PngDir) "s$i.png"), "PNG", $Width, $h) }
}
$deck.Close(); $pp.Quit()
