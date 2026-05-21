param(
    [string]$Model = "qwen2.5-1.5b",
    [ValidateSet("huggingface", "modelscope")]
    [string]$Provider = "huggingface",
    [string]$OutputDir = "hf_models",
    [switch]$SkipModel,
    [int]$LlmLayers = 6,
    [ValidateSet("skip", "cpu", "cu128", "cu130")]
    [string]$TorchBackend = "cu130"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is not installed or not on PATH. Install uv first: https://docs.astral.sh/uv/"
}

$env:UV_CACHE_DIR = Join-Path $ProjectRoot ".uv-cache"
Write-Host "Using project root: $ProjectRoot"
Write-Host "Using UV_CACHE_DIR: $env:UV_CACHE_DIR"

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

Write-Host "Syncing Python environment with uv..."
Invoke-Checked { uv sync --extra download }

if ($TorchBackend -ne "skip") {
    $TorchIndex = switch ($TorchBackend) {
        "cpu" { "https://download.pytorch.org/whl/cpu" }
        "cu128" { "https://download.pytorch.org/whl/cu128" }
        "cu130" { "https://download.pytorch.org/whl/cu130" }
    }
    Write-Host "Installing PyTorch backend '$TorchBackend' from $TorchIndex ..."
    Invoke-Checked { uv pip install --reinstall torch torchvision torchaudio --index-url $TorchIndex }
} else {
    Write-Host "Skipping explicit PyTorch backend installation."
}

if (-not $SkipModel) {
    Write-Host "Downloading model '$Model' from '$Provider'..."
    Invoke-Checked { uv run --no-sync python scripts/download_model.py `
        --model $Model `
        --provider $Provider `
        --output-dir $OutputDir `
        --llm-layers $LlmLayers }
} else {
    Write-Host "Skipping model download."
}

Write-Host "Running a quick compile check..."
Invoke-Checked { uv run --no-sync python -m compileall src scripts }

Write-Host "Checking PyTorch CUDA visibility..."
Invoke-Checked { uv run --no-sync python scripts/check_cuda.py }

Write-Host ""
Write-Host "Setup finished."
Write-Host "Try a smoke test:"
Write-Host "  uv run --no-sync python scripts/generate_toy_data.py --output data/toy/od.npy"
Write-Host "  uv run --no-sync python src/train.py --config configs/od_llm_tiny.yaml"
