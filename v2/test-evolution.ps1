$ErrorActionPreference = "Stop"
Push-Location "$PSScriptRoot"

try {
    Write-Host "Cleaning and building..." -ForegroundColor Cyan
    dotnet clean | Out-Null
    dotnet build | Out-Null

    Write-Host "Killing lingering testhost..." -ForegroundColor Cyan
    powershell -ExecutionPolicy Bypass -File "scripts/kill-testhost.ps1" | Out-Null

    Write-Host "Running fast tests (skipping integration)..." -ForegroundColor Cyan
    dotnet test --filter Category!=Slow --no-build

    Write-Host "Running offline eval harness..." -ForegroundColor Cyan
    dotnet test --filter OfflineEvalTests --no-build

    Write-Host "Done." -ForegroundColor Green
} finally {
    Pop-Location
}
