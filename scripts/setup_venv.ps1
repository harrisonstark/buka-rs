# Local venv with PyTorch CUDA 11.8. A GTX 1080 wants this wheel, and float32.
# Uses repo-local .tmp/ for pip temp files.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

$env:TMP = Join-Path (Get-Location) ".tmp"
$env:TEMP = $env:TMP
New-Item -ItemType Directory -Force -Path $env:TMP | Out-Null

if (-not (Test-Path .venv)) {
    python -m venv .venv
}

.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\pip.exe install torch --index-url https://download.pytorch.org/whl/cu118
.\.venv\Scripts\pip.exe install -e ".[dev]"

Write-Host ""
Write-Host "Done. Activate with:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "Then:"
Write-Host "  python -m scripts.smoke_gpu"
Write-Host "  pytest"
