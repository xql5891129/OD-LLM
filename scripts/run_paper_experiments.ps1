param(
    [string]$TransformerConfig = "configs/real_od_transformer.yaml",
    [string]$ODLLMConfig = "configs/real_od_qwen.yaml",
    [string]$BaselineSuite = "configs/suite_baselines.yaml",
    [string]$AblationSuite = "configs/suite_ablation.yaml",
    [switch]$SkipCudaCheck,
    [switch]$SkipBaselines,
    [switch]$SkipAblations,
    [switch]$DryRun
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

if ($DryRun) {
    Write-Host "Dry run: commands will be printed by suite runner where applicable."
    Write-Host "uv run --no-sync python src/train.py --config $TransformerConfig"
    Write-Host "uv run --no-sync python src/train.py --config $ODLLMConfig"
} else {
    Invoke-Checked { uv run --no-sync python src/train.py --config $TransformerConfig }
    Invoke-Checked { uv run --no-sync python src/train.py --config $ODLLMConfig }
}

if (-not $SkipBaselines) {
    if ($DryRun) {
        Invoke-Checked { uv run --no-sync python scripts/run_experiment_suite.py --suite $BaselineSuite --dry-run }
    } else {
        Invoke-Checked { uv run --no-sync python scripts/run_experiment_suite.py --suite $BaselineSuite }
    }
}

if (-not $SkipAblations) {
    if ($DryRun) {
        Invoke-Checked { uv run --no-sync python scripts/run_experiment_suite.py --suite $AblationSuite --dry-run }
    } else {
        Invoke-Checked { uv run --no-sync python scripts/run_experiment_suite.py --suite $AblationSuite }
    }
}

if (-not $DryRun) {
    Invoke-Checked { uv run --no-sync python src/compare.py --root outputs --output outputs/paper_experiments_comparison.csv }
    Write-Host "Saved final comparison: outputs/paper_experiments_comparison.csv"
}
