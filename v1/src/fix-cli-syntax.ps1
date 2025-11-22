#!/usr/bin/env pwsh

# Fix TARS CLI F# Syntax Errors
# Fixes specific syntax issues preventing compilation

Write-Host "🔧 Fixing TARS CLI F# Syntax Errors" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""

# Function to fix syntax errors in a file
function Fix-FileSyntax {
    param([string]$FilePath, [string]$Description)
    
    if (Test-Path $FilePath) {
        Write-Host "📝 Fixing $Description..." -ForegroundColor Yellow
        
        $content = Get-Content $FilePath -Raw
        $updated = $false
        
        # Fix incomplete System.Threading statements
        if ($content -match 'System\.Threading\.// REAL:') {
            $content = $content -replace 'System\.Threading\.// REAL:', 'System.Threading.Thread.Sleep(500) // REAL:'
            $updated = $true
            Write-Host "  ✅ Fixed incomplete System.Threading statements" -ForegroundColor Green
        }
        
        # Fix incomplete System.Threading statements (alternative pattern)
        if ($content -match 'System\.Threading\.\s*\r?\n') {
            $content = $content -replace 'System\.Threading\.\s*\r?\n', 'System.Threading.Thread.Sleep(500)' + [Environment]::NewLine
            $updated = $true
            Write-Host "  ✅ Fixed incomplete System.Threading statements (pattern 2)" -ForegroundColor Green
        }
        
        if ($updated) {
            Set-Content -Path $FilePath -Value $content -Encoding UTF8
            Write-Host "  📦 Syntax errors fixed in $Description" -ForegroundColor Green
        } else {
            Write-Host "  ℹ️  No syntax fixes needed in $Description" -ForegroundColor Blue
        }
        
        Write-Host ""
    } else {
        Write-Host "❌ File not found: $FilePath" -ForegroundColor Red
    }
}

# Files with reported syntax errors
$errorFiles = @(
    @{ Path = "TarsEngine.FSharp.Cli/Commands/ConsciousChatbotCommand.fs"; Description = "ConsciousChatbotCommand.fs" },
    @{ Path = "TarsEngine.FSharp.Cli/Commands/LiveDemoCommand.fs"; Description = "LiveDemoCommand.fs" },
    @{ Path = "TarsEngine.FSharp.Cli/Commands/QACommand.fs"; Description = "QACommand.fs" },
    @{ Path = "TarsEngine.FSharp.Cli/Agents/AutonomousModificationEngine.fs"; Description = "AutonomousModificationEngine.fs" }
)

foreach ($file in $errorFiles) {
    Fix-FileSyntax -FilePath $file.Path -Description $file.Description
}

Write-Host "📊 SYNTAX FIX SUMMARY" -ForegroundColor Cyan
Write-Host "=====================" -ForegroundColor Cyan
Write-Host "Files processed: $($errorFiles.Count)" -ForegroundColor White
Write-Host ""

Write-Host "🎯 Next Steps:" -ForegroundColor Yellow
Write-Host "1. Run 'dotnet build TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj' to test fixes" -ForegroundColor White
Write-Host "2. If errors persist, manually review and fix remaining syntax issues" -ForegroundColor White
Write-Host ""

Write-Host "✅ CLI syntax error fixes completed!" -ForegroundColor Green
