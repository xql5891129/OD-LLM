param(
    [ValidateSet("guangdianyuan", "daxuecheng", "jiefangbei")]
    [string]$Region = "guangdianyuan",
    [ValidateSet("30", "60")]
    [string]$Granularity = "60",
    [string[]]$Only = @("no_poi", "no_weather", "local_only", "no_sparse_hurdle"),
    [switch]$DryRun,
    [switch]$ContinueOnError,
    [switch]$SkipCompare
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$BaseConfig = "configs/line_bus/${Region}_${Granularity}_base.yaml"
$ArgsList = @(
    "-Suite", "configs/line_bus/suite_ablation.yaml",
    "-BaseConfig", $BaseConfig
)
if ($Only.Count -gt 0) {
    $ArgsList += "-Only"
    $ArgsList += $Only
}
$ArgsList += @("-RunTag", $Region)
if ($DryRun) { $ArgsList += "-DryRun" }
if ($ContinueOnError) { $ArgsList += "-ContinueOnError" }
if ($SkipCompare) { $ArgsList += "-SkipCompare" }

& (Join-Path $ProjectRoot "scripts/run_suite.ps1") @ArgsList
