#!/usr/bin/env pwsh

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   TARS Swarm Demo Launcher" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Change to script directory
Set-Location $PSScriptRoot

Write-Host "Building TARS Swarm Demo..." -ForegroundColor Yellow
$buildResult = dotnet build TarsSwarmDemo/TarsSwarmDemo.fsproj --verbosity quiet
$buildExitCode = $LASTEXITCODE

if ($buildExitCode -ne 0) {
    Write-Host ""
    Write-Host "❌ Build failed! Please check the errors above." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "✅ Build successful! Starting demo..." -ForegroundColor Green
Write-Host ""

# Check for command line arguments
$mode = $args[0]

switch ($mode) {
    "interactive" {
        Write-Host "🎯 Starting interactive demo..." -ForegroundColor Cyan
        Write-Host "Use arrow keys to navigate the menu" -ForegroundColor Gray
        Write-Host ""
        dotnet run --project TarsSwarmDemo/TarsSwarmDemo.fsproj -- interactive
    }
    "status" {
        Write-Host "📊 Checking swarm status..." -ForegroundColor Cyan
        dotnet run --project TarsSwarmDemo/TarsSwarmDemo.fsproj -- status
    }
    "test" {
        Write-Host "🧪 Running swarm tests..." -ForegroundColor Cyan
        dotnet run --project TarsSwarmDemo/TarsSwarmDemo.fsproj -- test
    }
    "monitor" {
        Write-Host "📈 Starting performance monitor..." -ForegroundColor Cyan
        dotnet run --project TarsSwarmDemo/TarsSwarmDemo.fsproj -- monitor
    }
    default {
        Write-Host "🚀 Running simple demo..." -ForegroundColor Cyan
        dotnet run --project TarsSwarmDemo/TarsSwarmDemo.fsproj
    }
}

Write-Host ""
Write-Host "Demo completed." -ForegroundColor Green
Write-Host ""
Write-Host "Available modes:" -ForegroundColor Yellow
Write-Host "  .\run-tars-swarm-demo.ps1 interactive  # Interactive menu" -ForegroundColor Gray
Write-Host "  .\run-tars-swarm-demo.ps1 status       # Show status only" -ForegroundColor Gray
Write-Host "  .\run-tars-swarm-demo.ps1 test         # Run tests only" -ForegroundColor Gray
Write-Host "  .\run-tars-swarm-demo.ps1 monitor      # Performance monitor" -ForegroundColor Gray
Write-Host ""
Read-Host "Press Enter to exit"
