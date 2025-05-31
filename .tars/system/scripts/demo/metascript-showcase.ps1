# TARS Metascript Showcase Script
# Demonstrates metascript capabilities by category

param(
    [string]$Category = "all",
    [switch]$Interactive = $false
)

Write-Host "📜 TARS Metascript Showcase" -ForegroundColor Cyan
Write-Host "===========================" -ForegroundColor Cyan
Write-Host ""

function Show-MetascriptCategory {
    param(
        [string]$CategoryName,
        [string]$Description,
        [string[]]$ExampleScripts
    )
    
    Write-Host "🎯 $CategoryName" -ForegroundColor Yellow
    Write-Host "   $Description" -ForegroundColor Gray
    Write-Host ""
    
    if ($ExampleScripts.Count -gt 0) {
        Write-Host "   📋 Example Scripts:" -ForegroundColor White
        foreach ($script in $ExampleScripts) {
            Write-Host "      • $script" -ForegroundColor Green
        }
        Write-Host ""
    }
}

function Demo-MetascriptDiscovery {
    Write-Host "🔍 Discovering All Available Metascripts..." -ForegroundColor Yellow
    Write-Host ""
    dotnet run --project TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj -- metascript-list --discover
    Write-Host ""
}

# Main showcase logic
Write-Host "🎯 Metascript Category Showcase" -ForegroundColor Green
Write-Host ""

if ($Category -eq "all" -or $Category -eq "core") {
    Show-MetascriptCategory -CategoryName "Core Metascripts" `
        -Description "Essential functionality and basic operations" `
        -ExampleScripts @("code_analysis.tars", "hello_world.tars", "simple_fsharp.tars")
}

if ($Category -eq "all" -or $Category -eq "autonomous") {
    Show-MetascriptCategory -CategoryName "Autonomous Metascripts" `
        -Description "Self-improvement and autonomous workflows" `
        -ExampleScripts @("autonomous_improvement.tars", "autonomous_improvement_enhanced.tars", "self_improvement_workflow.tars")
}

if ($Category -eq "all" -or $Category -eq "tree-of-thought") {
    Show-MetascriptCategory -CategoryName "Tree-of-Thought Metascripts" `
        -Description "Advanced reasoning and problem-solving patterns" `
        -ExampleScripts @("tree_of_thought_generator.tars", "ApplyImprovements.tars", "GenerateImprovements.tars", "CodeImprovement.tars")
}

if ($Category -eq "all" -or $Category -eq "docker") {
    Show-MetascriptCategory -CategoryName "Docker Integration Metascripts" `
        -Description "Container orchestration and Docker workflows" `
        -ExampleScripts @("docker_integration.tars", "docker_test.tars")
}

if ($Category -eq "all" -or $Category -eq "multi-agent") {
    Show-MetascriptCategory -CategoryName "Multi-Agent Metascripts" `
        -Description "Agent collaboration and coordination systems" `
        -ExampleScripts @("multi_agent_collaboration.tars", "tars_augment_collaboration.tars")
}

if ($Category -eq "all" -or $Category -eq "templates") {
    Show-MetascriptCategory -CategoryName "Template Metascripts" `
        -Description "Reusable templates for creating new metascripts" `
        -ExampleScripts @("basic-metascript.tars", "autonomous-improvement.tars")
}

Write-Host "🔍 **Live Metascript Discovery**" -ForegroundColor Magenta
Write-Host ""
Demo-MetascriptDiscovery

if ($Interactive) {
    Write-Host "🎮 Interactive Mode" -ForegroundColor Cyan
    Write-Host ""
    
    do {
        Write-Host "Select a category to explore:" -ForegroundColor Yellow
        Write-Host "  1. Core metascripts"
        Write-Host "  2. Autonomous metascripts"
        Write-Host "  3. Tree-of-Thought metascripts"
        Write-Host "  4. Docker integration"
        Write-Host "  5. Multi-agent systems"
        Write-Host "  6. Templates"
        Write-Host "  7. Discover all metascripts"
        Write-Host "  0. Exit"
        Write-Host ""
        
        $choice = Read-Host "Enter your choice (0-7)"
        
        switch ($choice) {
            "1" { & $MyInvocation.MyCommand.Path -Category "core" }
            "2" { & $MyInvocation.MyCommand.Path -Category "autonomous" }
            "3" { & $MyInvocation.MyCommand.Path -Category "tree-of-thought" }
            "4" { & $MyInvocation.MyCommand.Path -Category "docker" }
            "5" { & $MyInvocation.MyCommand.Path -Category "multi-agent" }
            "6" { & $MyInvocation.MyCommand.Path -Category "templates" }
            "7" { Demo-MetascriptDiscovery }
            "0" { break }
            default { Write-Host "Invalid choice. Please try again." -ForegroundColor Red }
        }
        
        if ($choice -ne "0") {
            Write-Host ""
            Read-Host "Press Enter to continue"
            Clear-Host
        }
        
    } while ($choice -ne "0")
}

Write-Host "✅ Metascript Showcase Completed!" -ForegroundColor Green
Write-Host ""
Write-Host "💡 **Metascript Statistics:**" -ForegroundColor Yellow
Write-Host "   • Total Categories: 6"
Write-Host "   • Organized Scripts: 17+"
Write-Host "   • Total Available: 164+"
Write-Host ""
Write-Host "🚀 **Next Steps:**" -ForegroundColor Yellow
Write-Host "   • Run specific metascripts with TARS CLI"
Write-Host "   • Create custom metascripts using templates"
Write-Host "   • Explore autonomous improvement workflows"
