#!/usr/bin/env pwsh

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   TARS CLI Swarm Demo" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Change to script directory
Set-Location $PSScriptRoot

Write-Host "Building TARS CLI..." -ForegroundColor Yellow
$buildResult = dotnet build TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj --verbosity quiet
$buildExitCode = $LASTEXITCODE

if ($buildExitCode -ne 0) {
    Write-Host ""
    Write-Host "❌ Build failed! Please check the errors above." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "✅ Build successful!" -ForegroundColor Green
Write-Host ""
Write-Host "🚀 Demonstrating TARS CLI Swarm Features..." -ForegroundColor Cyan
Write-Host ""

Write-Host "========================================" -ForegroundColor Yellow
Write-Host "1. Full Demo (Header + Status + Tests)" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow
dotnet run --project TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj -- swarm demo

Write-Host ""
Write-Host "========================================" -ForegroundColor Yellow
Write-Host "2. Container Status Only" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow
dotnet run --project TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj -- swarm status

Write-Host ""
Write-Host "========================================" -ForegroundColor Yellow
Write-Host "3. Health Tests Only" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow
dotnet run --project TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj -- swarm test

Write-Host ""
Write-Host "========================================" -ForegroundColor Yellow
Write-Host "4. Performance Monitor" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow
dotnet run --project TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj -- swarm monitor

Write-Host ""
Write-Host "========================================" -ForegroundColor Yellow
Write-Host "5. Command Execution Demo" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow
dotnet run --project TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj -- swarm commands

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "   Demo Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Available TARS CLI swarm commands:" -ForegroundColor Yellow
Write-Host "  tars swarm demo      # Full demo with header" -ForegroundColor Gray
Write-Host "  tars swarm status    # Container status table" -ForegroundColor Gray
Write-Host "  tars swarm test      # Health test results" -ForegroundColor Gray
Write-Host "  tars swarm monitor   # Performance monitoring" -ForegroundColor Gray
Write-Host "  tars swarm commands  # Command execution demo" -ForegroundColor Gray
Write-Host ""
Read-Host "Press Enter to exit"
