[CmdletBinding(PositionalBinding = $false)]
param(
    [ValidateSet("guangdianyuan", "daxuecheng", "jiefangbei")]
    [string]$Region = "guangdianyuan",
    [ValidateSet("30", "60")]
    [string]$Granularity = "60",
    [string]$Suite = "configs/line_bus/suite_baselines.yaml",
    [string[]]$Only = @("ha", "lstm", "transformer", "odmixer", "odcrn"),
    [switch]$ResumeMissing,
    [switch]$DryRun,
    [switch]$ContinueOnError,
    [switch]$SkipCompare,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$BaseConfig = "configs/line_bus/${Region}_${Granularity}_base.yaml"

function Normalize-ExperimentNames {
    param([string[]]$Names)
    $items = @()
    foreach ($name in $Names) {
        foreach ($part in ($name -split ",")) {
            $clean = $part.Trim()
            if ($clean -ne "") {
                $items += $clean
            }
        }
    }
    return $items
}

$Only = Normalize-ExperimentNames $Only
if ($ExtraArgs.Count -gt 0) {
    $extraOnly = @()
    foreach ($arg in $ExtraArgs) {
        switch ($arg) {
            "-ResumeMissing" { $ResumeMissing = $true }
            "-DryRun" { $DryRun = $true }
            "-ContinueOnError" { $ContinueOnError = $true }
            "-SkipCompare" { $SkipCompare = $true }
            default {
                if ($arg.StartsWith("-")) {
                    throw "Unknown argument after -Only: $arg"
                }
                $extraOnly += $arg
            }
        }
    }
    if ($extraOnly.Count -gt 0) {
        $Only += Normalize-ExperimentNames $extraOnly
    }
}
if ($ResumeMissing) {
    $OutputRoot = Join-Path $ProjectRoot ("outputs\line_bus\{0}_{1}min" -f $Region, $Granularity)
    $Missing = @()
    foreach ($name in $Only) {
        $MetricsPath = Join-Path $OutputRoot ("{0}\logs\test_metrics.json" -f $name)
        if (-not (Test-Path -LiteralPath $MetricsPath)) {
            $Missing += $name
        }
    }
    $Only = $Missing
}
if ($Only.Count -eq 0) {
    Write-Host "No baseline experiments to run."
    exit 0
}
Write-Host ("Selected baselines: {0}" -f ($Only -join ", "))

$ArgsList = @(
    "-Suite", $Suite,
    "-BaseConfig", $BaseConfig,
    "-RunTag", ("{0}_{1}min" -f $Region, $Granularity)
)
if ($Only.Count -gt 0) {
    $ArgsList += "-Only"
    $ArgsList += $Only
}
if ($DryRun) { $ArgsList += "-DryRun" }
if ($ContinueOnError) { $ArgsList += "-ContinueOnError" }
if ($SkipCompare) { $ArgsList += "-SkipCompare" }

& (Join-Path $ProjectRoot "scripts/run_suite.ps1") @ArgsList
