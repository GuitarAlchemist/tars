# TARS Demo Script
# Demonstrates core TARS functionality

param(
    [string]$DemoType = "basic",
    [switch]$Verbose = $false
)

Write-Host "🚀 TARS Demo Script" -ForegroundColor Cyan
Write-Host "===================" -ForegroundColor Cyan
Write-Host ""

# Set verbose preference
if ($Verbose) {
    $VerbosePreference = "Continue"
}

function Show-TarsVersion {
    Write-Host "📋 TARS Version Information:" -ForegroundColor Yellow
    dotnet run --project TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj -- version
    Write-Host ""
}

function Show-TarsHelp {
    Write-Host "❓ TARS Help Information:" -ForegroundColor Yellow
    dotnet run --project TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj -- help
    Write-Host ""
}

function Demo-MetascriptDiscovery {
    Write-Host "📜 TARS Metascript Discovery:" -ForegroundColor Yellow
    Write-Host "Discovering available metascripts..." -ForegroundColor Gray
    dotnet run --project TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj -- metascript-list --discover
    Write-Host ""
}

function Demo-IntelligenceMeasurement {
    Write-Host "🧠 TARS Intelligence Measurement:" -ForegroundColor Yellow
    Write-Host "Measuring AI intelligence metrics..." -ForegroundColor Gray
    dotnet run --project TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj -- intelligence measure
    Write-Host ""
}

function Demo-MLTraining {
    Write-Host "🤖 TARS ML Training:" -ForegroundColor Yellow
    Write-Host "Running ML model training simulation..." -ForegroundColor Gray
    dotnet run --project TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj -- ml train
    Write-Host ""
}

function Demo-CodeAnalysis {
    Write-Host "🔍 TARS Code Analysis:" -ForegroundColor Yellow
    Write-Host "Analyzing code quality and structure..." -ForegroundColor Gray
    dotnet run --project TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj -- analyze
    Write-Host ""
}

function Demo-Compilation {
    Write-Host "🔨 TARS Compilation:" -ForegroundColor Yellow
    Write-Host "Testing F# compilation capabilities..." -ForegroundColor Gray
    dotnet run --project TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj -- compile
    Write-Host ""
}

# Main demo logic
switch ($DemoType.ToLower()) {
    "basic" {
        Write-Host "🎯 Running Basic TARS Demo" -ForegroundColor Green
        Write-Host ""
        Show-TarsVersion
        Show-TarsHelp
        Demo-IntelligenceMeasurement
        Demo-MLTraining
    }
    
    "full" {
        Write-Host "🎯 Running Full TARS Demo" -ForegroundColor Green
        Write-Host ""
        Show-TarsVersion
        Show-TarsHelp
        Demo-MetascriptDiscovery
        Demo-IntelligenceMeasurement
        Demo-MLTraining
        Demo-CodeAnalysis
        Demo-Compilation
    }
    
    "metascripts" {
        Write-Host "🎯 Running Metascripts Demo" -ForegroundColor Green
        Write-Host ""
        Show-TarsVersion
        Demo-MetascriptDiscovery
    }
    
    "intelligence" {
        Write-Host "🎯 Running Intelligence Demo" -ForegroundColor Green
        Write-Host ""
        Show-TarsVersion
        Demo-IntelligenceMeasurement
        Demo-MLTraining
    }
    
    default {
        Write-Host "❌ Unknown demo type: $DemoType" -ForegroundColor Red
        Write-Host ""
        Write-Host "Available demo types:" -ForegroundColor Yellow
        Write-Host "  basic       - Basic TARS functionality"
        Write-Host "  full        - Complete TARS demonstration"
        Write-Host "  metascripts - Metascript discovery and listing"
        Write-Host "  intelligence - AI intelligence and ML features"
        Write-Host ""
        Write-Host "Usage: .\tars-demo.ps1 -DemoType <type> [-Verbose]"
        exit 1
    }
}

Write-Host "✅ TARS Demo Completed Successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "💡 Next Steps:" -ForegroundColor Yellow
Write-Host "  • Explore individual commands: dotnet run -- help"
Write-Host "  • Try metascripts: dotnet run -- metascript-list"
Write-Host "  • Run intelligence tests: dotnet run -- intelligence measure"
Write-Host "  • Check ML capabilities: dotnet run -- ml train"
