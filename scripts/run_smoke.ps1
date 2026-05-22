param(
    [switch]$SkipCudaCheck
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot
$env:UV_CACHE_DIR = Join-Path $ProjectRoot ".uv-cache"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [scriptblock]$Command
    )
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE`: $Command"
    }
}

if (-not $SkipCudaCheck) {
    Invoke-Checked { uv run --no-sync python scripts/check_cuda.py }
}

if (-not (Test-Path "data/toy/od.npy")) {
    Invoke-Checked { uv run --no-sync python scripts/generate_toy_data.py --output data/toy/od.npy }
}

Invoke-Checked { uv run --no-sync python src/train.py --config configs/default.yaml }
Invoke-Checked { uv run --no-sync python src/train.py --config configs/od_llm_tiny.yaml }
Invoke-Checked { uv run --no-sync python src/compare.py --root outputs --output outputs/comparison.csv }

Write-Host ""
Write-Host "Smoke run finished."
Write-Host "Metrics table: outputs/comparison.csv"
