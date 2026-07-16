$ErrorActionPreference = 'Continue'
Set-Location -Path $PSScriptRoot

Write-Output '=== Pass 1: pdflatex ==='
pdflatex -interaction=nonstopmode main.tex 2>&1 | Out-Null

Write-Output '=== Pass 2: bibtex ==='
$bibOut = bibtex main 2>&1
$bibOut | Select-Object -Last 20

Write-Output '=== Pass 3: pdflatex ==='
pdflatex -interaction=nonstopmode main.tex 2>&1 | Out-Null

Write-Output '=== Pass 4: pdflatex ==='
$lastOut = pdflatex -interaction=nonstopmode main.tex 2>&1

Write-Output '=== Warnings/Errors from final pass ==='
$lastOut | Select-String -Pattern 'Warning|Error|Undefined|Overfull|Underfull' | Select-Object -First 30

Write-Output '=== Final PDF ==='
if (Test-Path 'main.pdf') {
    Get-Item 'main.pdf' | Format-List Name, Length, LastWriteTime
} else {
    Write-Output 'PDF MANQUANT'
}
