$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"

function Get-BootstrapPython {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        return @("py", "-3")
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        return @("python")
    }
    throw "Python was not found. Install Python 3.9+ or Docker Desktop."
}

if (-not (Test-Path $venvPython)) {
    $bootstrap = Get-BootstrapPython
    Write-Host "Creating virtual environment..."
    if ($bootstrap.Length -eq 2) {
        & $bootstrap[0] $bootstrap[1] -m venv .venv
    } else {
        & $bootstrap[0] -m venv .venv
    }
}

Write-Host "Installing dependencies..."
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r requirements.txt

Write-Host "Downloading or verifying model files..."
& $venvPython -m app.download_models

Write-Host ""
Write-Host "Setup complete."
Write-Host "Run the app with:"
Write-Host "  powershell -ExecutionPolicy Bypass -File .\scripts\run-local.ps1"
