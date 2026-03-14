#!/usr/bin/env pwsh

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   TARS AUTONOMOUS UI EVOLUTION DEMO" -ForegroundColor Cyan  
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "🤖 TARS will now demonstrate autonomous UI creation and evolution" -ForegroundColor Green
Write-Host "📊 The system will analyze its own state and generate appropriate UI" -ForegroundColor Yellow
Write-Host "🔄 Watch as TARS creates and modifies its interface in real-time" -ForegroundColor Magenta
Write-Host ""
Read-Host "Press Enter to start the demonstration"

Write-Host ""
Write-Host "🚀 Starting TARS Autonomous UI Evolution..." -ForegroundColor Cyan
Write-Host ""

# Run the autonomous UI evolution metascript
Write-Host "📋 Executing: autonomous_ui_evolution.trsx" -ForegroundColor Yellow
Write-Host ""

try {
    & dotnet run --project "../../TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj" -- run ".tars/metascripts/autonomous_ui_evolution.trsx"
    
    Write-Host ""
    Write-Host "🌐 Opening evolved UI in browser..." -ForegroundColor Green
    Write-Host ""
    
    # Open the generated UI
    if ($IsWindows) {
        Start-Process ".tars/projects/tars_evolved_ui.html"
    } elseif ($IsMacOS) {
        & open ".tars/projects/tars_evolved_ui.html"
    } else {
        & xdg-open ".tars/projects/tars_evolved_ui.html"
    }
    
    Write-Host ""
    Write-Host "✅ Demo complete!" -ForegroundColor Green
    Write-Host ""
    Write-Host "🎯 What you just witnessed:" -ForegroundColor Cyan
    Write-Host "   • TARS analyzed its own system state" -ForegroundColor White
    Write-Host "   • Generated F# React components based on current needs" -ForegroundColor White
    Write-Host "   • Created responsive layouts dynamically" -ForegroundColor White
    Write-Host "   • Deployed the UI with hot reload capabilities" -ForegroundColor White
    Write-Host ""
    Write-Host "🔄 In a full implementation, this would happen continuously:" -ForegroundColor Magenta
    Write-Host "   • Every few seconds, TARS re-analyzes its state" -ForegroundColor White
    Write-Host "   • Components are added, modified, or removed as needed" -ForegroundColor White
    Write-Host "   • The UI evolves to match system requirements" -ForegroundColor White
    Write-Host "   • Users see a living, breathing interface that adapts" -ForegroundColor White
    Write-Host ""
    Write-Host "💡 Key Features Demonstrated:" -ForegroundColor Yellow
    Write-Host "   • Autonomous component generation" -ForegroundColor White
    Write-Host "   • Real-time UI adaptation" -ForegroundColor White
    Write-Host "   • Performance-based UI changes" -ForegroundColor White
    Write-Host "   • Agent activity monitoring" -ForegroundColor White
    Write-Host "   • Zero-downtime deployments" -ForegroundColor White
    Write-Host ""
    
} catch {
    Write-Host "❌ Error running demo: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
    Write-Host "💡 Note: This demo requires the TARS CLI to be built." -ForegroundColor Yellow
    Write-Host "   Run 'dotnet build' in the TarsEngine.FSharp.Cli directory first." -ForegroundColor Yellow
}

Read-Host "Press Enter to exit"
