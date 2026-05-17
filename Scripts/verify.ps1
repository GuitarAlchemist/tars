#requires -Version 7
<#
.SYNOPSIS
    Repo-level verification oracle for Agent Blackbox.
.DESCRIPTION
    Builds and tests the active tars solution (`v2/Tars.sln`). The Agent
    Blackbox workflow uses this as VERIFY_COMMAND. Exit code 0 means
    "harness is green". Anything non-zero means the loop / PR should be
    treated as failed.

    Writes a minimal state/quality/tars-harness/last.json so the
    overseer can pick up the metric on the next cycle.
.NOTES
    Pre-existing build failures on main (see CLAUDE.md / governance
    notes) are NOT something this script tries to fix — it just reports.
#>
[CmdletBinding()]
param(
    [switch]$BuildOnly,
    [switch]$NoState
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$status = 'ok'
$metricValue = 1.0
$summary = 'Harness verify succeeded.'

function Write-LastJson {
    param([string]$Status, [double]$MetricValue, [string]$Summary)
    if ($NoState) { return }
    $dir = Join-Path $repoRoot 'state/quality/tars-harness'
    if (-not (Test-Path -LiteralPath $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
    $payload = [ordered]@{
        domain          = 'tars-harness'
        emitted_at      = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
        metric_name     = 'harness_ready'
        metric_value    = $MetricValue
        oracle_status   = $Status
        oracle_command  = 'pwsh Scripts/verify.ps1'
        summary         = $Summary
    }
    $payload | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $dir 'last.json') -Encoding utf8
}

try {
    if (Test-Path -LiteralPath (Join-Path $repoRoot 'v2/Tars.sln')) {
        Push-Location (Join-Path $repoRoot 'v2')
        try {
            dotnet build Tars.sln
            if ($LASTEXITCODE -ne 0) { throw "dotnet build failed (exit $LASTEXITCODE)" }
            if (-not $BuildOnly) {
                dotnet test Tars.sln
                if ($LASTEXITCODE -ne 0) { throw "dotnet test failed (exit $LASTEXITCODE)" }
            }
        }
        finally {
            Pop-Location
        }
    }
    elseif (Test-Path -LiteralPath (Join-Path $repoRoot 'Tars.sln')) {
        dotnet build (Join-Path $repoRoot 'Tars.sln')
        if ($LASTEXITCODE -ne 0) { throw "dotnet build failed (exit $LASTEXITCODE)" }
    }
    else {
        throw 'No Tars.sln found under repo root or v2/.'
    }

    Write-LastJson -Status 'ok' -MetricValue 1.0 -Summary $summary
    exit 0
}
catch {
    $status = 'fail'
    $metricValue = 0.0
    $summary = $_.Exception.Message
    Write-Host "verify.ps1 FAILED: $summary" -ForegroundColor Red
    Write-LastJson -Status $status -MetricValue $metricValue -Summary $summary
    exit 1
}
