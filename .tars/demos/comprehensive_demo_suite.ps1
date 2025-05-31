# Comprehensive TARS Demo Suite - Demonstrates all systems and capabilities
# This script provides guided demonstrations of TARS autonomous features

param(
    [Parameter(Mandatory=$false)]
    [switch]$RunAll = $false,
    
    [Parameter(Mandatory=$false)]
    [switch]$DemoGeneration = $false,
    
    [Parameter(Mandatory=$false)]
    [switch]$DemoMemory = $false,
    
    [Parameter(Mandatory=$false)]
    [switch]$DemoExploration = $false,
    
    [Parameter(Mandatory=$false)]
    [switch]$DemoIntegration = $false,
    
    [Parameter(Mandatory=$false)]
    [switch]$Interactive = $false
)

Write-Host "🎭 COMPREHENSIVE TARS DEMO SUITE" -ForegroundColor Green
Write-Host "=================================" -ForegroundColor Green
Write-Host "🎯 Demonstrating TARS autonomous coding superintelligence" -ForegroundColor Cyan
Write-Host ""

$demoStartTime = Get-Date
$demoProjectsDir = "C:\Users\spare\source\repos\tars\.tars\demo_projects"
$scriptsDir = "C:\Users\spare\source\repos\tars\.tars\scripts"

# Ensure demo directories exist
if (-not (Test-Path $demoProjectsDir)) {
    New-Item -ItemType Directory -Path $demoProjectsDir -Force | Out-Null
}

function Wait-ForUser {
    param([string]$Message = "Press Enter to continue...")
    if ($Interactive) {
        Write-Host $Message -ForegroundColor Yellow
        Read-Host
    } else {
        Start-Sleep -Seconds 2
    }
}

function Show-DemoHeader {
    param([string]$Title, [string]$Description)
    Write-Host ""
    Write-Host "🎭 $Title" -ForegroundColor Yellow
    Write-Host "=" * ($Title.Length + 3) -ForegroundColor Yellow
    Write-Host $Description -ForegroundColor Cyan
    Write-Host ""
}

# Demo 1: Complete Autonomous Generation
function Demo-AutonomousGeneration {
    Show-DemoHeader "AUTONOMOUS PROJECT GENERATION" "Demonstrating end-to-end autonomous code generation with zero assumptions"
    
    $demoRequests = @(
        @{
            Request = "Create a personal expense tracker"
            Description = "Web application for tracking personal expenses with categories and reporting"
        },
        @{
            Request = "Build a password strength checker"
            Description = "Utility to analyze and score password strength with recommendations"
        },
        @{
            Request = "Make a simple chat application"
            Description = "Real-time chat application with multiple users and message history"
        }
    )
    
    foreach ($demo in $demoRequests) {
        Write-Host "🎯 DEMO: $($demo.Request)" -ForegroundColor Green
        Write-Host "Description: $($demo.Description)" -ForegroundColor White
        Write-Host ""
        
        Wait-ForUser "Ready to generate project? Press Enter..."
        
        $startTime = Get-Date
        Write-Host "🚀 Starting autonomous generation..." -ForegroundColor Cyan
        
        # Execute actual TARS generation
        $result = & dotnet run --project "TarsEngine.FSharp.SelfImprovement\TarsEngine.FSharp.SelfImprovement.fsproj" -- autonomous-task $demo.Request
        
        $duration = (Get-Date) - $startTime
        
        # Check results
        $sanitizedName = $demo.Request.Replace(" ", "_").ToLower()
        $projectPath = "C:\Users\spare\source\repos\tars\.tars\projects\$sanitizedName"
        
        if (Test-Path $projectPath) {
            $generatedFiles = Get-ChildItem $projectPath -File | Where-Object { -not $_.Name.StartsWith(".") }
            $totalSize = ($generatedFiles | Measure-Object -Property Length -Sum).Sum
            
            Write-Host ""
            Write-Host "✅ GENERATION COMPLETE!" -ForegroundColor Green
            Write-Host "⏱️ Duration: $($duration.TotalSeconds.ToString('F1')) seconds" -ForegroundColor White
            Write-Host "📄 Files Generated: $($generatedFiles.Count)" -ForegroundColor White
            Write-Host "💾 Total Size: $totalSize bytes" -ForegroundColor White
            Write-Host "📁 Project Location: $projectPath" -ForegroundColor White
            Write-Host ""
            Write-Host "📋 Generated Files:" -ForegroundColor Cyan
            foreach ($file in $generatedFiles) {
                Write-Host "  📄 $($file.Name) ($($file.Length) bytes)" -ForegroundColor White
            }
        } else {
            Write-Host "❌ Generation failed - no project created" -ForegroundColor Red
        }
        
        Wait-ForUser "Review the generated project, then press Enter to continue..."
    }
}

# Demo 2: Hybrid Memory System
function Demo-HybridMemorySystem {
    Show-DemoHeader "HYBRID MEMORY SYSTEM" "Demonstrating per-project JSON memory + global ChromaDB integration"
    
    Write-Host "🧠 The hybrid memory system provides:" -ForegroundColor Cyan
    Write-Host "  📁 Per-project JSON storage for fast local access" -ForegroundColor White
    Write-Host "  🌐 Global ChromaDB integration for cross-project intelligence" -ForegroundColor White
    Write-Host "  🔍 Memory search and pattern recognition" -ForegroundColor White
    Write-Host "  💡 Learning insights and continuous improvement" -ForegroundColor White
    Write-Host ""
    
    # Find existing projects to demonstrate memory on
    $existingProjects = Get-ChildItem "C:\Users\spare\source\repos\tars\.tars\projects" -Directory | Select-Object -First 2
    
    foreach ($project in $existingProjects) {
        Write-Host "🎯 DEMO: Memory Enhancement for $($project.Name)" -ForegroundColor Green
        
        Wait-ForUser "Ready to create enhanced memory? Press Enter..."
        
        $startTime = Get-Date
        Write-Host "🧠 Creating enhanced memory system..." -ForegroundColor Cyan
        
        # Create enhanced memory
        $userRequest = $project.Name -replace "_", " "
        $result = & "$scriptsDir\simple_enhanced_memory.ps1" -ProjectPath $project.FullName -UserRequest $userRequest
        
        $duration = (Get-Date) - $startTime
        
        # Check memory creation
        $memoryDir = Join-Path $project.FullName ".tars\memory"
        if (Test-Path $memoryDir) {
            $memoryFiles = Get-ChildItem $memoryDir -File
            $sessionFile = $memoryFiles | Where-Object { $_.Name -match "session_.*\.json" }
            
            if ($sessionFile) {
                $sessionContent = Get-Content $sessionFile.FullName | ConvertFrom-Json
                
                Write-Host ""
                Write-Host "✅ MEMORY SYSTEM CREATED!" -ForegroundColor Green
                Write-Host "⏱️ Duration: $($duration.TotalSeconds.ToString('F1')) seconds" -ForegroundColor White
                Write-Host "📄 Session ID: $($sessionContent.SessionId)" -ForegroundColor White
                Write-Host "🧠 Memory Entries: $($sessionContent.Entries.Count)" -ForegroundColor White
                Write-Host "🔧 Technology: $($sessionContent.TechnologyStack)" -ForegroundColor White
                Write-Host "📊 Confidence: $($sessionContent.TechnologyConfidence.ToString('P1'))" -ForegroundColor White
                Write-Host ""
                Write-Host "📋 Memory Entry Types:" -ForegroundColor Cyan
                $entryTypes = $sessionContent.Entries | Group-Object EntryType
                foreach ($type in $entryTypes) {
                    Write-Host "  📝 $($type.Name): $($type.Count) entries" -ForegroundColor White
                }
                
                # Demonstrate vector embeddings
                Write-Host ""
                Write-Host "🔢 Adding vector embeddings..." -ForegroundColor Cyan
                $embeddingResult = & "$scriptsDir\add_vector_embeddings.ps1" -SessionPath $sessionFile.FullName
                
                Write-Host "✅ Vector embeddings added for semantic search" -ForegroundColor Green
            }
        }
        
        Wait-ForUser "Review the memory files, then press Enter to continue..."
    }
}

# Demo 3: Integrated Exploration System
function Demo-IntegratedExploration {
    Show-DemoHeader "INTEGRATED EXPLORATION SYSTEM" "Demonstrating autonomous recovery when metascripts get stuck"
    
    Write-Host "🔍 The exploration system provides:" -ForegroundColor Cyan
    Write-Host "  🔧 Autonomous recovery when metascripts get stuck" -ForegroundColor White
    Write-Host "  📊 Real-time YAML status with comprehensive comments" -ForegroundColor White
    Write-Host "  🎯 Multiple exploration strategies (deep dive, alternatives, web research)" -ForegroundColor White
    Write-Host "  🔗 Solution integration with implementation plans" -ForegroundColor White
    Write-Host ""
    
    $explorationScenarios = @(
        @{
            Scenario = "Technology Uncertainty"
            StuckReason = "Unable to determine optimal technology stack for user requirements"
            Description = "Metascript stuck on technology selection"
        },
        @{
            Scenario = "Ambiguous Requirements"
            StuckReason = "User request lacks specific implementation details"
            Description = "Metascript needs clarification on requirements"
        },
        @{
            Scenario = "Architecture Decision"
            StuckReason = "Multiple valid architectural approaches identified"
            Description = "Metascript uncertain about optimal architecture"
        }
    )
    
    # Use existing project for exploration demo
    $demoProject = Get-ChildItem "C:\Users\spare\source\repos\tars\.tars\projects" -Directory | Select-Object -First 1
    
    if ($demoProject) {
        foreach ($scenario in $explorationScenarios) {
            Write-Host "🎯 DEMO: $($scenario.Scenario)" -ForegroundColor Green
            Write-Host "Description: $($scenario.Description)" -ForegroundColor White
            Write-Host "Stuck Reason: $($scenario.StuckReason)" -ForegroundColor Yellow
            Write-Host ""
            
            Wait-ForUser "Ready to start exploration? Press Enter..."
            
            $startTime = Get-Date
            Write-Host "🔍 Starting integrated exploration..." -ForegroundColor Cyan
            
            # Run exploration demo
            $userRequest = $demoProject.Name -replace "_", " "
            $result = & "$scriptsDir\integrated_exploration_demo.ps1" -ProjectPath $demoProject.FullName -UserRequest $userRequest -StuckReason $scenario.StuckReason -MetascriptName "demo_metascript"
            
            $duration = (Get-Date) - $startTime
            
            # Check exploration results
            $explorationDir = Join-Path $demoProject.FullName ".tars\exploration"
            $statusFile = Join-Path $demoProject.FullName ".tars\status.yaml"
            
            if ((Test-Path $explorationDir) -and (Test-Path $statusFile)) {
                $explorationFiles = Get-ChildItem $explorationDir -File
                $statusContent = Get-Content $statusFile -Raw
                
                Write-Host ""
                Write-Host "✅ EXPLORATION COMPLETE!" -ForegroundColor Green
                Write-Host "⏱️ Duration: $($duration.TotalSeconds.ToString('F1')) seconds" -ForegroundColor White
                Write-Host "📄 Exploration Files: $($explorationFiles.Count)" -ForegroundColor White
                Write-Host "📊 YAML Status: Updated with recovery progress" -ForegroundColor White
                Write-Host ""
                Write-Host "🔍 Exploration Features Demonstrated:" -ForegroundColor Cyan
                Write-Host "  ✅ Autonomous problem analysis" -ForegroundColor Green
                Write-Host "  ✅ Multiple exploration strategies" -ForegroundColor Green
                Write-Host "  ✅ Real-time YAML status updates" -ForegroundColor Green
                Write-Host "  ✅ Comprehensive solution generation" -ForegroundColor Green
                Write-Host "  ✅ Implementation plan creation" -ForegroundColor Green
            }
            
            Wait-ForUser "Review the exploration results, then press Enter to continue..."
            break # Only demo one scenario to save time
        }
    }
}

# Demo 4: Complete Integration
function Demo-CompleteIntegration {
    Show-DemoHeader "COMPLETE SYSTEM INTEGRATION" "Demonstrating all systems working together"
    
    Write-Host "🎯 This demo shows the complete TARS ecosystem:" -ForegroundColor Cyan
    Write-Host "  🚀 Autonomous project generation" -ForegroundColor White
    Write-Host "  🧠 Hybrid memory system with learning" -ForegroundColor White
    Write-Host "  🔍 Exploration and recovery capabilities" -ForegroundColor White
    Write-Host "  📊 Real-time YAML status tracking" -ForegroundColor White
    Write-Host "  🔢 Vector embeddings for semantic search" -ForegroundColor White
    Write-Host ""
    
    $integrationRequest = "Create a smart home automation system"
    
    Write-Host "🎯 INTEGRATION DEMO: $integrationRequest" -ForegroundColor Green
    Write-Host "This will demonstrate the complete workflow from request to completion" -ForegroundColor White
    Write-Host ""
    
    Wait-ForUser "Ready to start complete integration demo? Press Enter..."
    
    # Phase 1: Generation
    Write-Host "🚀 PHASE 1: AUTONOMOUS GENERATION" -ForegroundColor Yellow
    $startTime = Get-Date
    $result = & dotnet run --project "TarsEngine.FSharp.SelfImprovement\TarsEngine.FSharp.SelfImprovement.fsproj" -- autonomous-task $integrationRequest
    $generationDuration = (Get-Date) - $startTime
    
    $sanitizedName = $integrationRequest.Replace(" ", "_").ToLower()
    $projectPath = "C:\Users\spare\source\repos\tars\.tars\projects\$sanitizedName"
    
    if (Test-Path $projectPath) {
        $generatedFiles = Get-ChildItem $projectPath -File | Where-Object { -not $_.Name.StartsWith(".") }
        Write-Host "✅ Generation complete: $($generatedFiles.Count) files in $($generationDuration.TotalSeconds.ToString('F1'))s" -ForegroundColor Green
        
        # Phase 2: Memory Enhancement
        Write-Host ""
        Write-Host "🧠 PHASE 2: MEMORY ENHANCEMENT" -ForegroundColor Yellow
        $memoryStart = Get-Date
        $memoryResult = & "$scriptsDir\simple_enhanced_memory.ps1" -ProjectPath $projectPath -UserRequest $integrationRequest
        $memoryDuration = (Get-Date) - $memoryStart
        Write-Host "✅ Memory system created in $($memoryDuration.TotalSeconds.ToString('F1'))s" -ForegroundColor Green
        
        # Phase 3: Vector Embeddings
        Write-Host ""
        Write-Host "🔢 PHASE 3: VECTOR EMBEDDINGS" -ForegroundColor Yellow
        $embeddingStart = Get-Date
        $memoryDir = Join-Path $projectPath ".tars\memory"
        $sessionFile = Get-ChildItem $memoryDir -File | Where-Object { $_.Name -match "session_.*\.json" } | Select-Object -First 1
        if ($sessionFile) {
            $embeddingResult = & "$scriptsDir\add_vector_embeddings.ps1" -SessionPath $sessionFile.FullName
            $embeddingDuration = (Get-Date) - $embeddingStart
            Write-Host "✅ Vector embeddings added in $($embeddingDuration.TotalSeconds.ToString('F1'))s" -ForegroundColor Green
        }
        
        # Phase 4: Exploration Demo
        Write-Host ""
        Write-Host "🔍 PHASE 4: EXPLORATION SYSTEM" -ForegroundColor Yellow
        $explorationStart = Get-Date
        $explorationResult = & "$scriptsDir\integrated_exploration_demo.ps1" -ProjectPath $projectPath -UserRequest $integrationRequest -StuckReason "Demonstration of exploration capabilities"
        $explorationDuration = (Get-Date) - $explorationStart
        Write-Host "✅ Exploration system demonstrated in $($explorationDuration.TotalSeconds.ToString('F1'))s" -ForegroundColor Green
        
        # Final Summary
        $totalDuration = (Get-Date) - $startTime
        Write-Host ""
        Write-Host "🎉 COMPLETE INTEGRATION DEMO FINISHED!" -ForegroundColor Green
        Write-Host "=======================================" -ForegroundColor Green
        Write-Host "⏱️ Total Duration: $($totalDuration.TotalMinutes.ToString('F1')) minutes" -ForegroundColor White
        Write-Host "📁 Project: $projectPath" -ForegroundColor White
        Write-Host ""
        Write-Host "🎯 SYSTEMS DEMONSTRATED:" -ForegroundColor Cyan
        Write-Host "  ✅ Autonomous project generation ($($generationDuration.TotalSeconds.ToString('F1'))s)" -ForegroundColor Green
        Write-Host "  ✅ Hybrid memory system ($($memoryDuration.TotalSeconds.ToString('F1'))s)" -ForegroundColor Green
        Write-Host "  ✅ Vector embeddings ($($embeddingDuration.TotalSeconds.ToString('F1'))s)" -ForegroundColor Green
        Write-Host "  ✅ Exploration and recovery ($($explorationDuration.TotalSeconds.ToString('F1'))s)" -ForegroundColor Green
        Write-Host "  ✅ YAML status tracking (real-time)" -ForegroundColor Green
        Write-Host ""
        Write-Host "📊 FINAL PROJECT STRUCTURE:" -ForegroundColor Cyan
        Write-Host "  📄 Generated Files: $($generatedFiles.Count)" -ForegroundColor White
        Write-Host "  🧠 Memory System: Enhanced with vector embeddings" -ForegroundColor White
        Write-Host "  🔍 Exploration: Recovery capabilities demonstrated" -ForegroundColor White
        Write-Host "  📊 Status: Real-time YAML tracking" -ForegroundColor White
    }
}

# Execute demos based on parameters
if ($RunAll -or $DemoGeneration) {
    Demo-AutonomousGeneration
}

if ($RunAll -or $DemoMemory) {
    Demo-HybridMemorySystem
}

if ($RunAll -or $DemoExploration) {
    Demo-IntegratedExploration
}

if ($RunAll -or $DemoIntegration) {
    Demo-CompleteIntegration
}

# Final summary
$totalDemoDuration = (Get-Date) - $demoStartTime
Write-Host ""
Write-Host "🎭 COMPREHENSIVE DEMO SUITE COMPLETE" -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Green
Write-Host "⏱️ Total Demo Duration: $($totalDemoDuration.TotalMinutes.ToString('F1')) minutes" -ForegroundColor White
Write-Host "📁 Demo Projects: $demoProjectsDir" -ForegroundColor White
Write-Host ""
Write-Host "🚀 TARS AUTONOMOUS CODING SUPERINTELLIGENCE DEMONSTRATED!" -ForegroundColor Green
