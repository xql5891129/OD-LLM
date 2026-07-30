param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RawArgs
)

$ErrorActionPreference = "Stop"

$Suite = "configs/line_bus/suite_baselines.yaml"
$BaseConfig = ""
$Only = @()
$DryRun = $false
$ContinueOnError = $false
$SkipCompare = $false
$Python = ""
$LogDir = ""
$RunTag = ""

function Add-OnlyValues {
    param([string[]]$Values)
    foreach ($value in $Values) {
        foreach ($item in ($value -split ",")) {
            $name = $item.Trim()
            if ($name -ne "") {
                $script:Only += $name
            }
        }
    }
}

for ($i = 0; $i -lt $RawArgs.Count; $i++) {
    $arg = $RawArgs[$i]
    switch ($arg) {
        "-Suite" {
            $i += 1
            if ($i -ge $RawArgs.Count) { throw "-Suite requires a value" }
            $Suite = $RawArgs[$i]
        }
        "-BaseConfig" {
            $i += 1
            if ($i -ge $RawArgs.Count) { throw "-BaseConfig requires a value" }
            $BaseConfig = $RawArgs[$i]
        }
        "-Only" {
            $values = @()
            $i += 1
            while ($i -lt $RawArgs.Count -and -not $RawArgs[$i].StartsWith("-")) {
                $values += $RawArgs[$i]
                $i += 1
            }
            $i -= 1
            Add-OnlyValues $values
        }
        "-DryRun" {
            $DryRun = $true
        }
        "-ContinueOnError" {
            $ContinueOnError = $true
        }
        "-SkipCompare" {
            $SkipCompare = $true
        }
        "-Python" {
            $i += 1
            if ($i -ge $RawArgs.Count) { throw "-Python requires a value" }
            $Python = $RawArgs[$i]
        }
        "-LogDir" {
            $i += 1
            if ($i -ge $RawArgs.Count) { throw "-LogDir requires a value" }
            $LogDir = $RawArgs[$i]
        }
        "-RunTag" {
            $i += 1
            if ($i -ge $RawArgs.Count) { throw "-RunTag requires a value" }
            $RunTag = $RawArgs[$i]
        }
        default {
            throw "Unknown argument: $arg"
        }
    }
}

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot
$env:UV_CACHE_DIR = Join-Path $ProjectRoot ".uv-cache"

$ProjectPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if ($Python -eq "") {
    if (Test-Path -LiteralPath $ProjectPython) {
        $Python = $ProjectPython
    }
    else {
        $Python = "python"
    }
}

$ArgsList = @("scripts/run_experiment_suite.py", "--suite", $Suite)
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
if ($ContinueOnError) {
    $ArgsList += "--continue-on-error"
}
if ($SkipCompare) {
    $ArgsList += "--skip-compare"
}
$ArgsList += @("--python", $Python)
if ($LogDir -ne "") {
    $ArgsList += @("--log-dir", $LogDir)
}
if ($RunTag -ne "") {
    $ArgsList += @("--run-tag", $RunTag)
}

if (Get-Command uv -ErrorAction SilentlyContinue) {
    & uv @("run", "--no-sync", "python") @ArgsList
}
else {
    & $Python @ArgsList
}
if ($LASTEXITCODE -ne 0) {
    throw "Experiment suite failed with exit code $LASTEXITCODE"
}
