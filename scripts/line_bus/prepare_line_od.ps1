param(
    [ValidateSet("guangdianyuan", "daxuecheng", "jiefangbei")]
    [string]$Region = "guangdianyuan",
    [ValidateSet("30", "60")]
    [string]$Granularity = "60",
    [double]$MinTotalFlow = 100.0,
    [int]$MinStops = 4,
    [int]$PoiTopCategories = 30,
    [int]$MaxLines = 0,
    [int]$MaxCardFiles = 0,
    [string[]]$LineNos = @(),
    [switch]$AllRegions,
    [switch]$NoProgress
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $ProjectRoot
$env:UV_CACHE_DIR = Join-Path $ProjectRoot ".uv-cache"

$ScriptArgs = @(
    "data_processing/line_bus/prepare_line_od.py",
    "--interval-minutes", $Granularity,
    "--min-total-flow", "$MinTotalFlow",
    "--min-stops", "$MinStops",
    "--poi-top-categories", "$PoiTopCategories",
    "--use-card-line-dirs"
)
if (-not $AllRegions) {
    $ScriptArgs += @("--regions", $Region)
}
if ($MaxLines -gt 0) {
    $ScriptArgs += @("--max-lines", "$MaxLines")
}
if ($MaxCardFiles -gt 0) {
    $ScriptArgs += @("--max-card-files", "$MaxCardFiles")
}
if ($LineNos.Count -gt 0) {
    $ScriptArgs += "--line-nos"
    $ScriptArgs += $LineNos
}
if ($NoProgress) {
    $ScriptArgs += "--no-progress"
}

$ProjectPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (Get-Command uv -ErrorAction SilentlyContinue) {
    & uv @("run", "--no-sync", "python") @ScriptArgs
}
elseif (Test-Path -LiteralPath $ProjectPython) {
    & $ProjectPython @ScriptArgs
}
else {
    & python @ScriptArgs
}
if ($LASTEXITCODE -ne 0) {
    throw "Line OD preprocessing failed with exit code $LASTEXITCODE"
}
