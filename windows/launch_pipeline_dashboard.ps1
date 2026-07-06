$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptDir "..")).Path

if (-not $env:CONDA_BIN -or [string]::IsNullOrWhiteSpace($env:CONDA_BIN)) {
    $candidates = @(
        "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
        "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
        "C:\ProgramData\anaconda3\Scripts\conda.exe",
        "C:\ProgramData\miniconda3\Scripts\conda.exe"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            $env:CONDA_BIN = $candidate
            break
        }
    }
}

if (-not $env:CONDA_BIN -or [string]::IsNullOrWhiteSpace($env:CONDA_BIN)) {
    $env:CONDA_BIN = "conda"
}
if (-not $env:DASHBOARD_ENV -or [string]::IsNullOrWhiteSpace($env:DASHBOARD_ENV)) {
    $env:DASHBOARD_ENV = "dashboard_gui"
}

$condaCmd = Get-Command $env:CONDA_BIN -ErrorAction SilentlyContinue
if (-not $condaCmd -and -not (Test-Path $env:CONDA_BIN)) {
    Write-Host "Cannot find conda: $($env:CONDA_BIN)"
    Write-Host "Set CONDA_BIN=C:\path\to\conda.exe if needed."
    exit 2
}

Set-Location $repoRoot
Write-Host "Launching calcium pipeline dashboard..."
Write-Host "Repo root: $repoRoot"
Write-Host "Conda: $($env:CONDA_BIN)"
Write-Host "Environment: $($env:DASHBOARD_ENV)"

& $env:CONDA_BIN run -n $env:DASHBOARD_ENV --no-capture-output python current/pipeline_dashboard_gui.py @args
