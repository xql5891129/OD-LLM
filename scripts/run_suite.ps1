param(
    [string]$Suite = "configs/suite_baselines.yaml",
    [string]$BaseConfig = "",
    [string[]]$Only = @(),
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot
$env:UV_CACHE_DIR = Join-Path $ProjectRoot ".uv-cache"

$ArgsList = @("run", "--no-sync", "python", "scripts/run_experiment_suite.py", "--suite", $Suite)
if ($BaseConfig -ne "") {
    $ArgsList += @("--base-config", $BaseConfig)
}
if ($Only.Count -gt 0) {
    $ArgsList += "--only"
    $ArgsList += $Only
}
if ($DryRun) {
    $ArgsList += "--dry-run"
}

& uv @ArgsList
if ($LASTEXITCODE -ne 0) {
    throw "Experiment suite failed with exit code $LASTEXITCODE"
}
