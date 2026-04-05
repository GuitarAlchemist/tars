# TARS Specialized Agent Teams Demonstration Script
# Showcases the new specialized agent teams functionality

Write-Host "🤖 TARS SPECIALIZED AGENT TEAMS DEMONSTRATION" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
Write-Host ""

# Build the project first
Write-Host "🔨 Building TARS CLI with new teams functionality..." -ForegroundColor Yellow
dotnet build TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Build failed. Please check for compilation errors." -ForegroundColor Red
    exit 1
}

Write-Host "✅ Build successful!" -ForegroundColor Green
Write-Host ""

# Change to the CLI directory for execution
Set-Location "TarsEngine.FSharp.Cli"

Write-Host "📋 DEMONSTRATION SEQUENCE:" -ForegroundColor Cyan
Write-Host "1. List all available specialized teams" -ForegroundColor White
Write-Host "2. Show detailed information for key teams" -ForegroundColor White
Write-Host "3. Demonstrate team creation process" -ForegroundColor White
Write-Host "4. Run comprehensive teams demo" -ForegroundColor White
Write-Host ""

# Demo 1: List all available teams
Write-Host "🎬 DEMO 1: Listing All Available Specialized Teams" -ForegroundColor Yellow
Write-Host "=================================================" -ForegroundColor Yellow
dotnet run -- teams list
Write-Host ""
Read-Host "Press Enter to continue to next demo..."

# Demo 2: Show DevOps Team details
Write-Host "🎬 DEMO 2: DevOps Team Detailed Information" -ForegroundColor Yellow
Write-Host "===========================================" -ForegroundColor Yellow
dotnet run -- teams details "DevOps Team"
Write-Host ""
Read-Host "Press Enter to continue to next demo..."

# Demo 3: Show AI Team details
Write-Host "🎬 DEMO 3: AI Research Team Detailed Information" -ForegroundColor Yellow
Write-Host "===============================================" -ForegroundColor Yellow
dotnet run -- teams details "AI Team"
Write-Host ""
Read-Host "Press Enter to continue to next demo..."

# Demo 4: Show Technical Writers Team details
Write-Host "🎬 DEMO 4: Technical Writers Team Detailed Information" -ForegroundColor Yellow
Write-Host "=====================================================" -ForegroundColor Yellow
dotnet run -- teams details "Technical Writers Team"
Write-Host ""
Read-Host "Press Enter to continue to next demo..."

# Demo 5: Create a specific team
Write-Host "🎬 DEMO 5: Creating DevOps Team" -ForegroundColor Yellow
Write-Host "===============================" -ForegroundColor Yellow
dotnet run -- teams create "DevOps Team"
Write-Host ""
Read-Host "Press Enter to continue to next demo..."

# Demo 6: Create AI Team
Write-Host "🎬 DEMO 6: Creating AI Research Team" -ForegroundColor Yellow
Write-Host "====================================" -ForegroundColor Yellow
dotnet run -- teams create "AI Team"
Write-Host ""
Read-Host "Press Enter to continue to final demo..."

# Demo 7: Run comprehensive teams demo
Write-Host "🎬 DEMO 7: Comprehensive Teams Demonstration" -ForegroundColor Yellow
Write-Host "============================================" -ForegroundColor Yellow
Write-Host "This will demonstrate the creation of multiple teams simultaneously..." -ForegroundColor Cyan
dotnet run -- teams demo
Write-Host ""

# Demo 8: Show help
Write-Host "🎬 DEMO 8: Teams Command Help" -ForegroundColor Yellow
Write-Host "=============================" -ForegroundColor Yellow
dotnet run -- teams help
Write-Host ""

# Return to original directory
Set-Location ".."

Write-Host "🎉 DEMONSTRATION COMPLETED!" -ForegroundColor Green
Write-Host ""
Write-Host "📊 SUMMARY OF NEW CAPABILITIES:" -ForegroundColor Cyan
Write-Host "• 8 Specialized Agent Teams Available" -ForegroundColor White
Write-Host "  - DevOps Team (Infrastructure & Deployment)" -ForegroundColor Gray
Write-Host "  - Technical Writers Team (Documentation)" -ForegroundColor Gray
Write-Host "  - Architecture Team (System Design)" -ForegroundColor Gray
Write-Host "  - Direction Team (Strategic Planning)" -ForegroundColor Gray
Write-Host "  - Innovation Team (Research & Experimentation)" -ForegroundColor Gray
Write-Host "  - Machine Learning Team (AI/ML Development)" -ForegroundColor Gray
Write-Host "  - UX Team (User Experience Design)" -ForegroundColor Gray
Write-Host "  - AI Team (Advanced AI Research)" -ForegroundColor Gray
Write-Host ""
Write-Host "• 6+ New Agent Personas" -ForegroundColor White
Write-Host "  - DevOps Engineer" -ForegroundColor Gray
Write-Host "  - Documentation Architect" -ForegroundColor Gray
Write-Host "  - Product Strategist" -ForegroundColor Gray
Write-Host "  - ML Engineer" -ForegroundColor Gray
Write-Host "  - UX Director" -ForegroundColor Gray
Write-Host "  - AI Research Director" -ForegroundColor Gray
Write-Host ""
Write-Host "• 3 Specialized Team Metascripts" -ForegroundColor White
Write-Host "  - DevOps Orchestration (.tars/metascripts/teams/devops_orchestration.trsx)" -ForegroundColor Gray
Write-Host "  - AI Research Coordination (.tars/metascripts/teams/ai_research_coordination.trsx)" -ForegroundColor Gray
Write-Host "  - Technical Writing Coordination (.tars/metascripts/teams/technical_writing_coordination.trsx)" -ForegroundColor Gray
Write-Host ""
Write-Host "• New CLI Command: 'tars teams'" -ForegroundColor White
Write-Host "  - List available teams" -ForegroundColor Gray
Write-Host "  - Show team details" -ForegroundColor Gray
Write-Host "  - Create and deploy teams" -ForegroundColor Gray
Write-Host "  - Run team demonstrations" -ForegroundColor Gray
Write-Host ""

Write-Host "🚀 NEXT STEPS:" -ForegroundColor Yellow
Write-Host "1. Explore individual team capabilities with 'tars teams details <team>'" -ForegroundColor White
Write-Host "2. Create teams for your specific projects with 'tars teams create <team>'" -ForegroundColor White
Write-Host "3. Integrate teams with existing TARS workflows" -ForegroundColor White
Write-Host "4. Develop custom team configurations for your organization" -ForegroundColor White
Write-Host ""

Write-Host "💡 TIP: Use 'tars teams help' for complete command reference" -ForegroundColor Cyan
Write-Host ""
Write-Host "✨ TARS now supports enterprise-level multi-agent team coordination!" -ForegroundColor Green
