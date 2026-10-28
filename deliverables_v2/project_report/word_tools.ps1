param(
    [Parameter(Mandatory = $true)][ValidateSet("convert", "update", "pdf")][string]$Mode,
    [Parameter(Mandatory = $true)][string]$In,
    [string]$Out = "",
    [string]$Pdf = ""
)
# Drives Microsoft Word by COM.
#   convert: .doc -> .docx (the supplied template, unchanged original)
#   update : open a .docx, update every field (contents, lists of figures and tables, page numbers),
#            save it as $Out (.docx) and optionally export $Pdf
#   pdf    : export $In to $Pdf without changing it
$ErrorActionPreference = "Stop"
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    if ($Mode -eq "convert") {
        $doc = $word.Documents.Open($In, $false, $true)
        $doc.SaveAs2($Out, 16)
        $doc.Close($false)
    }
    elseif ($Mode -eq "pdf") {
        $doc = $word.Documents.Open($In, $false, $true)
        $doc.ExportAsFixedFormat($Pdf, 17)
        $doc.Close($false)
    }
    else {
        $doc = $word.Documents.Open($In, $false, $false)
        $word.Options.UpdateFieldsAtPrint = $false
        # two passes: the first fixes the page numbers that the lists themselves shift
        for ($pass = 0; $pass -lt 2; $pass++) {
            $doc.Fields.Update() | Out-Null
            foreach ($t in $doc.TablesOfContents) { $t.Update() }
            foreach ($t in $doc.TablesOfFigures) { $t.Update() }
            $doc.Repaginate()
        }
        foreach ($s in $doc.Sections) {
            foreach ($f in $s.Footers) { $f.Range.Fields.Update() | Out-Null }
            foreach ($h in $s.Headers) { $h.Range.Fields.Update() | Out-Null }
        }
        $doc.SaveAs2($Out, 16)
        if ($Pdf -ne "") { $doc.ExportAsFixedFormat($Pdf, 17) }
        "pages: " + $doc.ComputeStatistics(2)
        $doc.Close($false)
    }
}
finally {
    $word.Quit()
}
