# Upload bundle for Editorial Manager.
# Runs _compile.ps1, then produces submission\ (flat, no subfolder, as JIIS requires)
# and submission.zip with:
#   main.tex, sn-jnl.cls, sn-basic.bst, references.bib, main.bbl, Fig1..Fig7.pdf, main.pdf
$ErrorActionPreference = 'Continue'
Set-Location -Path $PSScriptRoot

& .\_compile.ps1

$out = 'submission'
if (Test-Path $out) { Remove-Item $out -Recurse -Force }
New-Item -ItemType Directory -Path $out | Out-Null

$files = @('main.tex', 'sn-jnl.cls', 'sn-basic.bst', 'references.bib', 'main.bbl', 'main.pdf') + (1..7 | ForEach-Object { "Fig$_.pdf" })
Copy-Item -Path $files -Destination $out

if (Test-Path 'submission.zip') { Remove-Item 'submission.zip' -Force }
Compress-Archive -Path "$out\*" -DestinationPath 'submission.zip'
Write-Output '=== submission.zip ==='
Get-ChildItem $out | Format-Table Name, Length
