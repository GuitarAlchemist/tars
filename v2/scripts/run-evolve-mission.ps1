param(
    [int]$MaxIterations = 3,
    [decimal]$BudgetUsd = 5.0,
    [string]$Model = "",
    [switch]$Verbose,
    [switch]$Quiet
)

$ErrorActionPreference = 'Stop'

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$reportDir = Join-Path -Path (Get-Location) -ChildPath "mission_reports"
if (-not (Test-Path $reportDir)) {
    New-Item -ItemType Directory -Path $reportDir | Out-Null
}

$logPath = Join-Path $reportDir ("evolve_mission_{0}.log" -f $timestamp)

$args = @("evolve", "--max-iterations", $MaxIterations.ToString(), "--budget", $BudgetUsd.ToString())
if ($Verbose) { $args += "--verbose" }
if ($Quiet) { $args += "--quiet" }
if (-not [string]::IsNullOrWhiteSpace($Model)) {
    $args += @("--model", $Model)
}

Write-Host "Running evolve mission..." -ForegroundColor Cyan
Write-Host ("Args: {0}" -f ($args -join " "))

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = "dotnet"
$psi.Arguments = "run --project src/Tars.Interface.Cli -- " + ($args -join " ")
$psi.WorkingDirectory = (Get-Location).Path
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
$psi.UseShellExecute = $false
$psi.CreateNoWindow = $true

$proc = New-Object System.Diagnostics.Process
$proc.StartInfo = $psi

$null = $proc.Start()
$stdout = $proc.StandardOutput.ReadToEnd()
$stderr = $proc.StandardError.ReadToEnd()
$proc.WaitForExit()

$stdout | Out-File -FilePath $logPath -Encoding UTF8
if (-not [string]::IsNullOrWhiteSpace($stderr)) {
    "`n=== STDERR ===`n" | Out-File -FilePath $logPath -Append -Encoding UTF8
    $stderr | Out-File -FilePath $logPath -Append -Encoding UTF8
}

Write-Host "Mission log written to $logPath" -ForegroundColor Green
Write-Host ("Exit code: {0}" -f $proc.ExitCode)

if ($proc.ExitCode -ne 0) {
    Write-Warning "Evolve mission exited with non-zero code. See log for details."
}
