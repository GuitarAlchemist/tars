#!/usr/bin/env pwsh

# Fix CLI Application Errors
# Addresses the 4 main CLI application interface issues

Write-Host "🔧 Fixing TARS CLI Application Errors" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

$cliAppPath = "TarsEngine.FSharp.Cli/Core/CliApplication.fs"

if (Test-Path $cliAppPath) {
    Write-Host "📝 Reading CLI application file..." -ForegroundColor Yellow
    
    $content = Get-Content $cliAppPath -Raw
    $updated = $false
    
    Write-Host "🔧 Fixing ExecuteAsync signature..." -ForegroundColor Yellow
    
    # Fix ExecuteAsync signature - add options parameter
    if ($content -match 'member _\.ExecuteAsync\(args\) =') {
        $content = $content -replace 'member _\.ExecuteAsync\(args\) =', 'member _.ExecuteAsync(args) (options) ='
        $updated = $true
        Write-Host "  ✅ Fixed ExecuteAsync signature" -ForegroundColor Green
    }
    
    Write-Host "🔧 Adding missing Usage member..." -ForegroundColor Yellow
    
    # Add Usage member
    if ($content -match 'member _\.Description = "TARS Superintelligence - Real Tier 2/3 autonomous modification"') {
        $content = $content -replace 'member _\.Description = "TARS Superintelligence - Real Tier 2/3 autonomous modification"', 'member _.Description = "TARS Superintelligence - Real Tier 2/3 autonomous modification"' + [Environment]::NewLine + '                    member _.Usage = "tars superintelligence [evolve|assess]"'
        $updated = $true
        Write-Host "  ✅ Added Usage member" -ForegroundColor Green
    }
    
    Write-Host "🔧 Fixing return type..." -ForegroundColor Yellow
    
    # Fix return type from int to CommandResult
    if ($content -match 'return 0') {
        $content = $content -replace 'return 0', 'return CommandResult.success "Superintelligence command completed"'
        $updated = $true
        Write-Host "  ✅ Fixed return type" -ForegroundColor Green
    }
    
    if ($updated) {
        Set-Content -Path $cliAppPath -Value $content -Encoding UTF8
        Write-Host "  📦 CLI application updated successfully" -ForegroundColor Green
    } else {
        Write-Host "  ℹ️  No updates needed or patterns not found" -ForegroundColor Blue
    }
    
    Write-Host ""
} else {
    Write-Host "❌ CLI application file not found: $cliAppPath" -ForegroundColor Red
    exit 1
}

Write-Host "🎯 Next Steps:" -ForegroundColor Yellow
Write-Host "1. Test build: dotnet build TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj" -ForegroundColor White
Write-Host "2. Fix remaining agent errors if needed" -ForegroundColor White
Write-Host ""

Write-Host "✅ CLI Application fixes applied!" -ForegroundColor Green
