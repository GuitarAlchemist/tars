#!/usr/bin/env pwsh

# Fix TARS CLI Project File
# Directly fixes escaped package references and version issues

Write-Host "🔧 Fixing TARS CLI Project File" -ForegroundColor Cyan
Write-Host "===============================" -ForegroundColor Cyan
Write-Host ""

$projectPath = "TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj"

if (Test-Path $projectPath) {
    Write-Host "📝 Reading project file..." -ForegroundColor Yellow
    
    $content = Get-Content $projectPath -Raw
    
    Write-Host "🔧 Applying fixes..." -ForegroundColor Yellow
    
    # Fix all escaped package references and versions in one go
    $content = $content -replace 'Microsoft\\\.Extensions\\\.DependencyInjection" Version="10\.0\.0"', 'Microsoft.Extensions.DependencyInjection" Version="9.0.0"'
    $content = $content -replace 'Microsoft\\\.Extensions\\\.Http" Version="10\.0\.0"', 'Microsoft.Extensions.Http" Version="9.0.0"'
    $content = $content -replace 'Microsoft\\\.Extensions\\\.Logging" Version="10\.0\.0"', 'Microsoft.Extensions.Logging" Version="9.0.0"'
    $content = $content -replace 'Microsoft\\\.Extensions\\\.Hosting" Version="10\.0\.0"', 'Microsoft.Extensions.Hosting" Version="9.0.0"'
    $content = $content -replace 'System\\\.Threading\\\.Channels" Version="10\.0\.0"', 'System.Threading.Channels" Version="9.0.0"'
    $content = $content -replace 'System\\\.Text\\\.Json" Version="10\.0\.0"', 'System.Text.Json" Version="9.0.0"'
    $content = $content -replace 'System\\\.ServiceProcess\\\.ServiceController" Version="10\.0\.0"', 'System.ServiceProcess.ServiceController" Version="9.0.0"'
    $content = $content -replace 'System\\\.Management" Version="10\.0\.0"', 'System.Management" Version="9.0.0"'
    
    Write-Host "💾 Writing fixed project file..." -ForegroundColor Yellow
    Set-Content -Path $projectPath -Value $content -Encoding UTF8
    
    Write-Host "✅ Project file fixed successfully!" -ForegroundColor Green
    
    Write-Host ""
    Write-Host "🔍 Verifying fixes..." -ForegroundColor Yellow
    
    # Verify the fixes
    $verifyContent = Get-Content $projectPath -Raw
    $escapedCount = ($verifyContent | Select-String -Pattern 'Include="[^"]*\\\.').Matches.Count
    $version10Count = ($verifyContent | Select-String -Pattern 'Version="10\.0\.0"').Matches.Count
    
    Write-Host "Escaped package references remaining: $escapedCount" -ForegroundColor $(if ($escapedCount -eq 0) { "Green" } else { "Red" })
    Write-Host "Version 10.0.0 references remaining: $version10Count" -ForegroundColor $(if ($version10Count -eq 0) { "Green" } else { "Red" })
    
} else {
    Write-Host "❌ Project file not found: $projectPath" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "🎯 Next Steps:" -ForegroundColor Yellow
Write-Host "1. Run 'dotnet restore TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj'" -ForegroundColor White
Write-Host "2. Run 'dotnet build TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj'" -ForegroundColor White
Write-Host ""

Write-Host "✅ CLI project file fixes completed!" -ForegroundColor Green
