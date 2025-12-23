param(
    [switch]$SkipStatus,
    [switch]$SkipTests,
    [switch]$SkipDemo,
    [switch]$SkipDiag
)

$ErrorActionPreference = 'Stop'

function Run-Step {
    param(
        [string]$Name,
        [string]$Command
    )

    Write-Host "==== $Name ====" -ForegroundColor Cyan
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        $output = Invoke-Expression $Command | Out-String
        $sw.Stop()
        Write-Host $output
        return @{ name = $Name; success = $true; durationMs = $sw.ElapsedMilliseconds; output = $output }
    }
    catch {
        $sw.Stop()
        Write-Error $_
        return @{ name = $Name; success = $false; durationMs = $sw.ElapsedMilliseconds; output = $_.ToString() }
    }
}

$results = @()

if (-not $SkipStatus) {
    $results += Run-Step -Name "Status" -Command "dotnet run --project src/Tars.Interface.Cli -- status"
}

if (-not $SkipTests) {
    $results += Run-Step -Name "Test Suite" -Command "dotnet test Tars.sln -v minimal"
}

if (-not $SkipDemo) {
    # Demo ping validates basic event bus + CLI wiring without needing external LLM
    $results += Run-Step -Name "Demo Ping" -Command "dotnet run --project src/Tars.Interface.Cli -- demo-ping"
}

if (-not $SkipDiag) {
    # CLI diagnostics (lightweight, no LLM)
    $results += Run-Step -Name "Diagnostics" -Command "dotnet run --project src/Tars.Interface.Cli -- diag"
}

# Summary
Write-Host "`n==== Summary ====" -ForegroundColor Green
foreach ($r in $results) {
    $status = if ($r.success) { "OK" } else { "FAIL" }
    Write-Host ("{0,-20} {1,4}  {2,6} ms" -f $r.name, $status, $r.durationMs)
}

# Write proof report to file
$reportPath = Join-Path -Path (Get-Location) -ChildPath "proof_report.txt"
"TARS Senior SWE Proof Run $(Get-Date -Format 'u')" | Out-File -FilePath $reportPath -Encoding UTF8
foreach ($r in $results) {
    $status = if ($r.success) { "OK" } else { "FAIL" }
    "=== $($r.name) [$status] ($($r.durationMs) ms) ===" | Out-File -FilePath $reportPath -Append -Encoding UTF8
    $r.output | Out-File -FilePath $reportPath -Append -Encoding UTF8
}

Write-Host "`nReport written to $reportPath" -ForegroundColor Green
