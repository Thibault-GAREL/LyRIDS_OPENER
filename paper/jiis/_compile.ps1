# Build of the JIIS paper.
# 1) builds references.bib = ../references.bib + verified DOIs (doi_overlay.csv),
# 2) compiles the standalone figures Fig3 to Fig6,
# 3) pdflatex -> bibtex -> pdflatex x2, then prints warnings and the page count (limit: 25).
$ErrorActionPreference = 'Continue'
Set-Location -Path $PSScriptRoot

Write-Output '=== references.bib (shared bib + DOI overlay) ==='
& 'c:\0-Code_py_temp\basic_env\Scripts\python.exe' _sync_bib.py

Write-Output '=== Figures (standalone) ==='
foreach ($n in 3, 4, 5, 6) {
    Push-Location 'figures_src'
    pdflatex -interaction=nonstopmode "Fig$n.tex" 2>&1 | Out-Null
    Pop-Location
    if (Test-Path "figures_src\Fig$n.pdf") {
        Copy-Item "figures_src\Fig$n.pdf" "Fig$n.pdf" -Force
        Write-Output "  Fig$n.pdf OK"
    } else {
        Write-Output "  Fig$n.pdf MANQUANT (voir figures_src\Fig$n.log)"
    }
}

Write-Output '=== Pass 1: pdflatex ==='
pdflatex -interaction=nonstopmode main.tex 2>&1 | Out-Null

Write-Output '=== Pass 2: bibtex ==='
bibtex main 2>&1 | Select-Object -Last 10

Write-Output '=== Pass 3: pdflatex ==='
pdflatex -interaction=nonstopmode main.tex 2>&1 | Out-Null

Write-Output '=== Pass 4: pdflatex ==='
pdflatex -interaction=nonstopmode main.tex 2>&1 | Out-Null

Write-Output '=== Warnings/Errors from final pass ==='
Select-String -Path 'main.log' -Pattern '^!|Undefined|undefined|Overfull' | Select-Object -First 30

Write-Output '=== Page count (JIIS limit: 25) ==='
Select-String -Path 'main.log' -Pattern 'Output written on main.pdf'
