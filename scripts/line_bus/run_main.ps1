param(
    [ValidateSet("guangdianyuan", "daxuecheng", "jiefangbei")]
    [string]$Region = "guangdianyuan",
    [ValidateSet("30", "60")]
    [string]$Granularity = "60",
    [switch]$DryRun,
    [switch]$SkipCompare
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$BaseConfig = "configs/line_bus/${Region}_${Granularity}_base.yaml"
$ArgsList = @(
    "-Suite", "configs/line_bus/suite_main.yaml",
    "-BaseConfig", $BaseConfig,
    "-Only", "od_llm"
)
if ($DryRun) { $ArgsList += "-DryRun" }
if ($SkipCompare) { $ArgsList += "-SkipCompare" }

& (Join-Path $ProjectRoot "scripts/run_suite.ps1") @ArgsList
