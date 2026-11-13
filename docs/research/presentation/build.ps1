param([string]$Soffice = "soffice.com")
$ErrorActionPreference = "Stop"
uv run --no-project --with python-docx==1.2.0 --with python-pptx==1.0.2 python (Join-Path $PSScriptRoot 'build.py')
if ($LASTEXITCODE -ne 0) { throw 'Editable document build failed.' }
foreach ($name in @('Research_Report.docx', 'Research_Presentation.pptx')) {
    $inputPath = Join-Path $PSScriptRoot $name
    $exportArguments = @('--headless', '--convert-to', 'pdf', '--outdir', ('"' + $PSScriptRoot + '"'), ('"' + $inputPath + '"'))
    $export = Start-Process -FilePath $Soffice -ArgumentList $exportArguments -WindowStyle Hidden -Wait -PassThru
    if ($export.ExitCode -ne 0) { throw "PDF export failed for $name" }
    $outputPath = [IO.Path]::ChangeExtension($inputPath, '.pdf')
    if (-not (Test-Path -LiteralPath $outputPath)) { throw "Missing PDF for $name" }
}
