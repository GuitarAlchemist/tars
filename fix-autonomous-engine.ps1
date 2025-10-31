#!/usr/bin/env pwsh

# Fix AutonomousModificationEngine.fs Syntax Errors
# Fixes specific syntax issues preventing compilation

Write-Host "🔧 Fixing AutonomousModificationEngine.fs Syntax Errors" -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host ""

$filePath = "TarsEngine.FSharp.Cli/Agents/AutonomousModificationEngine.fs"

if (Test-Path $filePath) {
    Write-Host "📝 Reading file..." -ForegroundColor Yellow
    
    $content = Get-Content $filePath -Raw
    $updated = $false
    
    Write-Host "🔧 Applying fixes..." -ForegroundColor Yellow
    
    # Fix incomplete do! statement
    if ($content -match 'do! // REAL:') {
        $content = $content -replace 'do! // REAL:', 'do! Task.Delay(100) // REAL:'
        $updated = $true
        Write-Host "  ✅ Fixed incomplete do! statement" -ForegroundColor Green
    }
    
    # Fix incomplete do! statement (alternative pattern)
    if ($content -match 'do!\s*\r?\n') {
        $content = $content -replace 'do!\s*\r?\n', 'do! Task.Delay(100)' + [Environment]::NewLine
        $updated = $true
        Write-Host "  ✅ Fixed incomplete do! statement (pattern 2)" -ForegroundColor Green
    }
    
    if ($updated) {
        Set-Content -Path $filePath -Value $content -Encoding UTF8
        Write-Host "  📦 Syntax errors fixed" -ForegroundColor Green
    } else {
        Write-Host "  ℹ️  No syntax fixes needed" -ForegroundColor Blue
    }
    
    Write-Host ""
} else {
    Write-Host "❌ File not found: $filePath" -ForegroundColor Red
    exit 1
}

Write-Host "🎯 Next Steps:" -ForegroundColor Yellow
Write-Host "1. Run 'dotnet build TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj' to test fixes" -ForegroundColor White
Write-Host ""

Write-Host "✅ AutonomousModificationEngine.fs syntax fixes completed!" -ForegroundColor Green
