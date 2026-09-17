# Upload bundle for the JIIS submission system (Springer Nature).
# "LaTeX documents with figures and tables compressed into a .zip format. We will compile these
# into a PDF for peer review."
# Runs _compile.ps1, then produces:
#   submission\                    flat folder (no subfolder), what goes in the zip
#   OPENER_JIIS_manuscript.zip     the file to upload in "Upload manuscript"
# Zip content: main.tex, sn-jnl.cls, sn-basic.bst, references.bib, main.bbl, Fig1..Fig7.pdf
# main.pdf is NOT in the zip: the journal compiles the sources itself, and a PDF inside the
# archive could be mistaken for the manuscript. Keep main.pdf locally to compare with their build.
$ErrorActionPreference = 'Continue'
Set-Location -Path $PSScriptRoot

& .\_compile.ps1

$out = 'submission'
if (Test-Path $out) { Remove-Item $out -Recurse -Force }
New-Item -ItemType Directory -Path $out | Out-Null

$files = @('main.tex', 'sn-jnl.cls', 'sn-basic.bst', 'references.bib', 'main.bbl') + (1..7 | ForEach-Object { "Fig$_.pdf" })
Copy-Item -Path $files -Destination $out

$zip = 'OPENER_JIIS_manuscript.zip'
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path "$out\*" -DestinationPath $zip
Write-Output "=== $zip ==="
Get-ChildItem $out | Format-Table Name, Length
