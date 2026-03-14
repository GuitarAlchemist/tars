# TARS Agent Scaling and Team Management Demo
# Demonstrates how to scale agent instances and manage specialized teams

Write-Host "🚀👥 TARS AGENT SCALING & TEAM MANAGEMENT DEMO" -ForegroundColor Green
Write-Host "===============================================" -ForegroundColor Green
Write-Host ""

# Define all available specialized teams
$SpecializedTeams = @{
    "Architecture Team" = @{
        description = "System design and architectural planning specialists"
        defaultSize = 4
        maxSize = 8
        personas = @("Architect", "Senior Developer", "Technical Lead")
        capabilities = @("system_design", "architectural_reviews", "technical_standards")
        color = "Blue"
    }
    "DevOps Team" = @{
        description = "Infrastructure, deployment, and operations specialists"
        defaultSize = 3
        maxSize = 6
        personas = @("DevOps Engineer", "Guardian", "Optimizer")
        capabilities = @("ci_cd", "monitoring", "infrastructure")
        color = "Green"
    }
    "Code Review Team" = @{
        description = "Expert code reviewers focused on quality and security"
        defaultSize = 4
        maxSize = 8
        personas = @("Senior Code Reviewer", "Technical Lead", "Guardian")
        capabilities = @("code_review", "security_analysis", "quality_assurance")
        color = "Yellow"
    }
    "Senior Development Team" = @{
        description = "Experienced developers with technical leadership"
        defaultSize = 4
        maxSize = 10
        personas = @("Senior Developer", "Technical Lead", "Architect")
        capabilities = @("advanced_development", "technical_leadership", "mentoring")
        color = "Cyan"
    }
    "Product Management Team" = @{
        description = "Product strategy and requirements management"
        defaultSize = 3
        maxSize = 6
        personas = @("Product Manager", "Product Strategist", "Communicator")
        capabilities = @("product_strategy", "requirements_management", "stakeholder_coordination")
        color = "Magenta"
    }
    "Project Management Team" = @{
        description = "Project coordination and delivery management"
        defaultSize = 3
        maxSize = 6
        personas = @("Project Manager", "Communicator", "Optimizer")
        capabilities = @("project_coordination", "resource_management", "delivery_optimization")
        color = "DarkYellow"
    }
    "Quality Assurance Team" = @{
        description = "Quality assurance and testing strategy"
        defaultSize = 4
        maxSize = 8
        personas = @("QA Lead", "Guardian", "Senior Code Reviewer")
        capabilities = @("testing_strategy", "quality_processes", "automation")
        color = "Red"
    }
    "Technical Writers Team" = @{
        description = "Documentation and knowledge management specialists"
        defaultSize = 3
        maxSize = 6
        personas = @("Documentation Architect", "Communicator", "Researcher")
        capabilities = @("technical_documentation", "knowledge_management", "user_guides")
        color = "DarkGreen"
    }
    "Innovation Team" = @{
        description = "Research, experimentation, and breakthrough solutions"
        defaultSize = 4
        maxSize = 8
        personas = @("Innovator", "Researcher", "AI Research Director")
        capabilities = @("research", "prototyping", "emerging_technologies")
        color = "DarkCyan"
    }
    "Machine Learning Team" = @{
        description = "AI/ML development and deployment specialists"
        defaultSize = 4
        maxSize = 8
        personas = @("ML Engineer", "Researcher", "AI Research Director")
        capabilities = @("ml_development", "mlops", "ai_ethics")
        color = "DarkMagenta"
    }
}

# Initialize agent scaling system
function Initialize-AgentScaling {
    Write-Host "🔧 Initializing TARS Agent Scaling System..." -ForegroundColor Cyan
    
    # Create teams directory
    $teamsDir = ".tars\teams"
    if (-not (Test-Path $teamsDir)) {
        New-Item -ItemType Directory -Path $teamsDir -Force | Out-Null
    }
    
    # Initialize team registry
    $global:teamRegistry = @{}
    $global:agentInstances = @{}
    $global:scalingMetrics = @{
        totalAgents = 0
        activeTeams = 0
        totalCapacity = 0
        utilizationRate = 0.0
        lastScalingAction = (Get-Date)
    }
    
    Write-Host "  ✅ Agent scaling system initialized" -ForegroundColor Green
    Write-Host ""
}

# Create and scale a team
function New-ScaledTeam {
    param(
        [string]$TeamName,
        [int]$DesiredSize = 0,
        [string]$ScalingReason = "Manual scaling"
    )
    
    $teamConfig = $SpecializedTeams[$TeamName]
    if (-not $teamConfig) {
        Write-Host "  ❌ Team '$TeamName' not found" -ForegroundColor Red
        return
    }
    
    # Determine team size
    $teamSize = if ($DesiredSize -gt 0) { 
        [Math]::Min($DesiredSize, $teamConfig.maxSize) 
    } else { 
        $teamConfig.defaultSize 
    }
    
    Write-Host "🚀 Creating $TeamName (Size: $teamSize)" -ForegroundColor $teamConfig.color
    Write-Host "  📝 $($teamConfig.description)" -ForegroundColor Gray
    Write-Host "  🎯 Reason: $ScalingReason" -ForegroundColor Gray
    
    # Create agent instances
    $teamAgents = @()
    for ($i = 1; $i -le $teamSize; $i++) {
        $personaIndex = ($i - 1) % $teamConfig.personas.Count
        $persona = $teamConfig.personas[$personaIndex]
        $agentId = "$($TeamName.Replace(' ', ''))_Agent_$i"
        
        $agent = @{
            id = $agentId
            teamName = $TeamName
            persona = $persona
            capabilities = $teamConfig.capabilities
            status = "active"
            workload = 0.0
            createdAt = (Get-Date)
            lastActivity = (Get-Date)
        }
        
        $teamAgents += $agent
        $global:agentInstances[$agentId] = $agent
        
        Write-Host "    🤖 Created: $agentId ($persona)" -ForegroundColor White
    }
    
    # Register team
    $team = @{
        name = $TeamName
        description = $teamConfig.description
        agents = $teamAgents
        currentSize = $teamSize
        maxSize = $teamConfig.maxSize
        defaultSize = $teamConfig.defaultSize
        capabilities = $teamConfig.capabilities
        createdAt = (Get-Date)
        lastScaled = (Get-Date)
        scalingHistory = @(@{
            action = "created"
            fromSize = 0
            toSize = $teamSize
            reason = $ScalingReason
            timestamp = (Get-Date)
        })
    }
    
    $global:teamRegistry[$TeamName] = $team
    
    # Update metrics
    $global:scalingMetrics.totalAgents += $teamSize
    $global:scalingMetrics.activeTeams += 1
    $global:scalingMetrics.totalCapacity += $teamSize * 100
    $global:scalingMetrics.lastScalingAction = Get-Date
    
    Write-Host "  ✅ Team created successfully with $teamSize agents" -ForegroundColor Green
    Write-Host ""
}

# Scale existing team up or down
function Set-TeamScale {
    param(
        [string]$TeamName,
        [int]$NewSize,
        [string]$ScalingReason = "Manual scaling"
    )
    
    $team = $global:teamRegistry[$TeamName]
    if (-not $team) {
        Write-Host "  ❌ Team '$TeamName' not found" -ForegroundColor Red
        return
    }
    
    $teamConfig = $SpecializedTeams[$TeamName]
    $currentSize = $team.currentSize
    $maxSize = $teamConfig.maxSize
    
    # Validate new size
    if ($NewSize -gt $maxSize) {
        Write-Host "  ⚠️ Requested size ($NewSize) exceeds maximum ($maxSize). Scaling to maximum." -ForegroundColor Yellow
        $NewSize = $maxSize
    }
    
    if ($NewSize -eq $currentSize) {
        Write-Host "  ℹ️ Team '$TeamName' already at requested size ($NewSize)" -ForegroundColor Gray
        return
    }
    
    Write-Host "📈 Scaling ${TeamName}: $currentSize → $NewSize agents" -ForegroundColor $teamConfig.color
    Write-Host "  🎯 Reason: $ScalingReason" -ForegroundColor Gray
    
    if ($NewSize -gt $currentSize) {
        # Scale up - add agents
        $agentsToAdd = $NewSize - $currentSize
        Write-Host "  ⬆️ Scaling UP: Adding $agentsToAdd agents" -ForegroundColor Green
        
        for ($i = $currentSize + 1; $i -le $NewSize; $i++) {
            $personaIndex = ($i - 1) % $teamConfig.personas.Count
            $persona = $teamConfig.personas[$personaIndex]
            $agentId = "$($TeamName.Replace(' ', ''))_Agent_$i"
            
            $agent = @{
                id = $agentId
                teamName = $TeamName
                persona = $persona
                capabilities = $teamConfig.capabilities
                status = "active"
                workload = 0.0
                createdAt = (Get-Date)
                lastActivity = (Get-Date)
            }
            
            $team.agents += $agent
            $global:agentInstances[$agentId] = $agent
            
            Write-Host "    ➕ Added: $agentId ($persona)" -ForegroundColor Green
        }
        
        $global:scalingMetrics.totalAgents += $agentsToAdd
        $global:scalingMetrics.totalCapacity += $agentsToAdd * 100
    }
    else {
        # Scale down - remove agents
        $agentsToRemove = $currentSize - $NewSize
        Write-Host "  ⬇️ Scaling DOWN: Removing $agentsToRemove agents" -ForegroundColor Yellow
        
        # Remove agents with lowest workload first
        $agentsToRemoveList = $team.agents | Sort-Object workload | Select-Object -First $agentsToRemove
        
        foreach ($agent in $agentsToRemoveList) {
            $team.agents = $team.agents | Where-Object { $_.id -ne $agent.id }
            $global:agentInstances.Remove($agent.id)
            Write-Host "    ➖ Removed: $($agent.id) ($($agent.persona))" -ForegroundColor Yellow
        }
        
        $global:scalingMetrics.totalAgents -= $agentsToRemove
        $global:scalingMetrics.totalCapacity -= $agentsToRemove * 100
    }
    
    # Update team
    $team.currentSize = $NewSize
    $team.lastScaled = Get-Date
    $team.scalingHistory += @{
        action = if ($NewSize -gt $currentSize) { "scaled_up" } else { "scaled_down" }
        fromSize = $currentSize
        toSize = $NewSize
        reason = $ScalingReason
        timestamp = (Get-Date)
    }
    
    $global:scalingMetrics.lastScalingAction = Get-Date
    
    Write-Host "  ✅ Team scaled successfully to $NewSize agents" -ForegroundColor Green
    Write-Host ""
}

# Show team status and scaling metrics
function Show-TeamStatus {
    param([string]$TeamName = "")
    
    Write-Host ""
    Write-Host "📊 TARS AGENT TEAM STATUS" -ForegroundColor Cyan
    Write-Host "=========================" -ForegroundColor Cyan
    Write-Host ""
    
    if ($TeamName) {
        # Show specific team
        $team = $global:teamRegistry[$TeamName]
        if ($team) {
            Show-SingleTeamStatus -Team $team
        } else {
            Write-Host "❌ Team '$TeamName' not found" -ForegroundColor Red
        }
    } else {
        # Show all teams
        foreach ($teamName in $global:teamRegistry.Keys | Sort-Object) {
            $team = $global:teamRegistry[$teamName]
            Show-SingleTeamStatus -Team $team
        }
        
        # Show overall metrics
        Write-Host "📈 Overall Scaling Metrics:" -ForegroundColor Yellow
        Write-Host "  Total Agents: $($global:scalingMetrics.totalAgents)" -ForegroundColor White
        Write-Host "  Active Teams: $($global:scalingMetrics.activeTeams)" -ForegroundColor White
        Write-Host "  Total Capacity: $($global:scalingMetrics.totalCapacity)" -ForegroundColor White
        Write-Host "  Last Scaling: $($global:scalingMetrics.lastScalingAction.ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor White
        Write-Host ""
    }
}

# Show single team status
function Show-SingleTeamStatus {
    param($Team)
    
    $teamConfig = $SpecializedTeams[$Team.name]
    
    Write-Host "🏢 $($Team.name)" -ForegroundColor $teamConfig.color
    Write-Host "  📝 $($Team.description)" -ForegroundColor Gray
    Write-Host "  👥 Size: $($Team.currentSize)/$($Team.maxSize) (Default: $($Team.defaultSize))" -ForegroundColor White
    Write-Host "  🎯 Capabilities: $($Team.capabilities -join ', ')" -ForegroundColor Gray
    Write-Host "  📅 Created: $($Team.createdAt.ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor Gray
    Write-Host "  🔄 Last Scaled: $($Team.lastScaled.ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor Gray
    
    Write-Host "  🤖 Agents:" -ForegroundColor White
    foreach ($agent in $Team.agents) {
        $workloadColor = if ($agent.workload -gt 80) { "Red" } elseif ($agent.workload -gt 60) { "Yellow" } else { "Green" }
        Write-Host "    • $($agent.id) ($($agent.persona)) - Workload: $($agent.workload)%" -ForegroundColor $workloadColor
    }
    Write-Host ""
}

# Demonstrate auto-scaling based on workload
function Demo-AutoScaling {
    Write-Host "🤖 DEMONSTRATING AUTO-SCALING CAPABILITIES" -ForegroundColor Cyan
    Write-Host "===========================================" -ForegroundColor Cyan
    Write-Host ""
    
    # Simulate high workload scenario
    Write-Host "📈 Scenario: High Development Workload" -ForegroundColor Yellow
    Write-Host "  • Large feature development project started" -ForegroundColor Gray
    Write-Host "  • Multiple code reviews needed" -ForegroundColor Gray
    Write-Host "  • Quality assurance requirements increased" -ForegroundColor Gray
    Write-Host ""
    
    # Scale up critical teams
    Set-TeamScale -TeamName "Senior Development Team" -NewSize 8 -ScalingReason "High development workload"
    Set-TeamScale -TeamName "Code Review Team" -NewSize 6 -ScalingReason "Increased code review demand"
    Set-TeamScale -TeamName "Quality Assurance Team" -NewSize 6 -ScalingReason "Enhanced quality requirements"
    
    Write-Host "📉 Scenario: Project Completion and Maintenance Phase" -ForegroundColor Yellow
    Write-Host "  • Feature development completed" -ForegroundColor Gray
    Write-Host "  • Reduced development activity" -ForegroundColor Gray
    Write-Host "  • Focus shifted to maintenance" -ForegroundColor Gray
    Write-Host ""
    
    # Scale down to normal levels
    Set-TeamScale -TeamName "Senior Development Team" -NewSize 4 -ScalingReason "Project completion"
    Set-TeamScale -TeamName "Code Review Team" -NewSize 4 -ScalingReason "Normal review workload"
    Set-TeamScale -TeamName "Quality Assurance Team" -NewSize 4 -ScalingReason "Maintenance phase"
}

# Main demo function
function Start-AgentScalingDemo {
    Write-Host "🚀👥 TARS Agent Scaling Demo Started" -ForegroundColor Green
    Write-Host ""
    Write-Host "💡 This demo shows how to:" -ForegroundColor Yellow
    Write-Host "  • Create specialized teams with multiple agents" -ForegroundColor White
    Write-Host "  • Scale teams up and down based on demand" -ForegroundColor White
    Write-Host "  • Manage different agent personas and capabilities" -ForegroundColor White
    Write-Host "  • Track scaling metrics and team performance" -ForegroundColor White
    Write-Host ""
    Write-Host "Commands: 'create', 'scale', 'status', 'auto-scale', 'help', 'exit'" -ForegroundColor Gray
    Write-Host ""
    
    $isRunning = $true
    while ($isRunning) {
        Write-Host ""
        $userInput = Read-Host "Command"
        
        switch ($userInput.ToLower().Trim()) {
            "exit" {
                $isRunning = $false
                Write-Host ""
                Write-Host "🚀👥 Agent scaling demo completed! Teams are ready for production." -ForegroundColor Green
                break
            }
            "create" {
                Write-Host ""
                Write-Host "Available teams:" -ForegroundColor Yellow
                $SpecializedTeams.Keys | Sort-Object | ForEach-Object { Write-Host "  • $_" -ForegroundColor White }
                Write-Host ""
                $teamName = Read-Host "Enter team name to create"
                if ($SpecializedTeams.ContainsKey($teamName)) {
                    New-ScaledTeam -TeamName $teamName -ScalingReason "User requested creation"
                } else {
                    Write-Host "❌ Invalid team name" -ForegroundColor Red
                }
            }
            "scale" {
                if ($global:teamRegistry.Count -eq 0) {
                    Write-Host "❌ No teams created yet. Use 'create' command first." -ForegroundColor Red
                    continue
                }
                Write-Host ""
                Write-Host "Active teams:" -ForegroundColor Yellow
                $global:teamRegistry.Keys | Sort-Object | ForEach-Object { Write-Host "  • $_" -ForegroundColor White }
                Write-Host ""
                $teamName = Read-Host "Enter team name to scale"
                if ($global:teamRegistry.ContainsKey($teamName)) {
                    $newSize = Read-Host "Enter new team size"
                    if ($newSize -match '^\d+$') {
                        Set-TeamScale -TeamName $teamName -NewSize ([int]$newSize) -ScalingReason "User requested scaling"
                    } else {
                        Write-Host "❌ Invalid size. Please enter a number." -ForegroundColor Red
                    }
                } else {
                    Write-Host "❌ Team not found" -ForegroundColor Red
                }
            }
            "status" {
                Show-TeamStatus
            }
            "auto-scale" {
                Demo-AutoScaling
            }
            "help" {
                Write-Host ""
                Write-Host "🚀👥 Agent Scaling Demo Commands:" -ForegroundColor Cyan
                Write-Host "• 'create' - Create a new specialized team" -ForegroundColor White
                Write-Host "• 'scale' - Scale an existing team up or down" -ForegroundColor White
                Write-Host "• 'status' - Show all team statuses and metrics" -ForegroundColor White
                Write-Host "• 'auto-scale' - Demonstrate automatic scaling scenarios" -ForegroundColor White
                Write-Host "• 'exit' - End the demo" -ForegroundColor White
            }
            default {
                Write-Host "Unknown command. Type 'help' for available commands." -ForegroundColor Red
            }
        }
    }
}

# Initialize and start
Initialize-AgentScaling

# Create some initial teams to demonstrate
Write-Host "🏗️ Creating Initial Teams for Demonstration..." -ForegroundColor Cyan
New-ScaledTeam -TeamName "Senior Development Team" -DesiredSize 4 -ScalingReason "Initial team setup"
New-ScaledTeam -TeamName "Code Review Team" -DesiredSize 4 -ScalingReason "Initial team setup"
New-ScaledTeam -TeamName "Product Management Team" -DesiredSize 3 -ScalingReason "Initial team setup"
New-ScaledTeam -TeamName "Project Management Team" -DesiredSize 3 -ScalingReason "Initial team setup"

Show-TeamStatus
Start-AgentScalingDemo
