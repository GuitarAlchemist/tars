#!/usr/bin/env pwsh

# Test script to capture MoE output
Write-Host "Testing TARS MoE System..." -ForegroundColor Green

# Test 1: Status
Write-Host "`n=== TEST 1: MoE Status ===" -ForegroundColor Yellow
dotnet run --project TarsEngine.FSharp.Cli -- moe status | Tee-Object -FilePath "moe_status_output.txt"

# Test 2: Simple Analysis
Write-Host "`n=== TEST 2: Simple Task Analysis ===" -ForegroundColor Yellow
dotnet run --project TarsEngine.FSharp.Cli -- moe analyze "Write a hello world program" | Tee-Object -FilePath "moe_simple_analysis.txt"

# Test 3: Complex Analysis
Write-Host "`n=== TEST 3: Complex Task Analysis ===" -ForegroundColor Yellow
dotnet run --project TarsEngine.FSharp.Cli -- moe analyze "Create a sophisticated machine learning system with advanced reasoning, mathematical optimization, and multilingual support" | Tee-Object -FilePath "moe_complex_analysis.txt"

# Test 4: Architecture
Write-Host "`n=== TEST 4: MoE Architecture ===" -ForegroundColor Yellow
dotnet run --project TarsEngine.FSharp.Cli -- moe architecture | Tee-Object -FilePath "moe_architecture.txt"

Write-Host "`n=== Tests Complete! Check output files ===" -ForegroundColor Green
Write-Host "Files created:" -ForegroundColor Cyan
Write-Host "- moe_status_output.txt" -ForegroundColor White
Write-Host "- moe_simple_analysis.txt" -ForegroundColor White
Write-Host "- moe_complex_analysis.txt" -ForegroundColor White
Write-Host "- moe_architecture.txt" -ForegroundColor White
