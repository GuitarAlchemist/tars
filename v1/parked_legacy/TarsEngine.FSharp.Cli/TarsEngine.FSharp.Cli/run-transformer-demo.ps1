#!/usr/bin/env pwsh

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   TARS Transformer Demo Launcher" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Choose a demo:" -ForegroundColor Yellow
Write-Host "  1. Model Recommendations (default)" -ForegroundColor White
Write-Host "  2. Performance Metrics" -ForegroundColor White
Write-Host "  3. Exit" -ForegroundColor White
Write-Host ""

$choice = Read-Host "Enter your choice (1-3)"

switch ($choice) {
    "2" {
        Write-Host ""
        Write-Host "Starting Performance Metrics Demo..." -ForegroundColor Green
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "dotnet run transformer perf"
    }
    "3" {
        Write-Host ""
        Write-Host "Goodbye!" -ForegroundColor Yellow
        exit 0
    }
    default {
        Write-Host ""
        Write-Host "Starting Model Recommendations Demo..." -ForegroundColor Green
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "dotnet run transformer models"
    }
}

Write-Host ""
Write-Host "Demo launched in new window!" -ForegroundColor Cyan
Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
