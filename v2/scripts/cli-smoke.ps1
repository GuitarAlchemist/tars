param(
    [string]$Model = "qwen2.5-coder:7b",
    [int]$EvolveIterations = 1,
    [decimal]$BudgetUsd = 1.0
)

$ErrorActionPreference = 'Stop'

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$reportDir = Join-Path -Path (Get-Location) -ChildPath "cli_smoke_reports"
if (-not (Test-Path $reportDir)) {
    New-Item -ItemType Directory -Path $reportDir | Out-Null
}

$logPath = Join-Path $reportDir ("cli_smoke_{0}.log" -f $timestamp)

function Run-Step {
    param(
        [string]$Name,
        [string]$Command
    )

    Write-Host "== $Name ==" -ForegroundColor Cyan
    Write-Host $Command -ForegroundColor DarkGray

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = "pwsh"
    $psi.Arguments = "-NoProfile -Command ""$Command"""
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

    "`n== $Name ==`n" | Out-File -FilePath $logPath -Append -Encoding UTF8
    $stdout | Out-File -FilePath $logPath -Append -Encoding UTF8
    if (-not [string]::IsNullOrWhiteSpace($stderr)) {
        "`n-- STDERR --`n" | Out-File -FilePath $logPath -Append -Encoding UTF8
        $stderr | Out-File -FilePath $logPath -Append -Encoding UTF8
    }

    if ($proc.ExitCode -ne 0) {
        Write-Warning "$Name failed with exit code $($proc.ExitCode). See log."
    } else {
        Write-Host "$Name OK" -ForegroundColor Green
    }
}

Run-Step -Name "Status" -Command "dotnet run --project src/Tars.Interface.Cli -- status"
Run-Step -Name "Diagnostics" -Command "dotnet run --project src/Tars.Interface.Cli -- diag"
Run-Step -Name "Demo Ping" -Command "dotnet run --project src/Tars.Interface.Cli -- demo-ping"
Run-Step -Name "Macro Demo" -Command "dotnet run --project src/Tars.Interface.Cli -- macro-demo"
Run-Step -Name "Evolve Smoke" -Command ("dotnet run --project src/Tars.Interface.Cli -- evolve --max-iterations {0} --budget {1} --model ""{2}"" --no-graphiti" -f $EvolveIterations, $BudgetUsd, $Model)

Write-Host "Smoke log written to $logPath" -ForegroundColor Yellow
