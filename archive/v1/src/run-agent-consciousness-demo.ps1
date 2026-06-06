# TARS Agent-Consciousness Integration Demo
# Demonstrates how agents participate in global consciousness state

Write-Host "🤖🧠 TARS AGENT-CONSCIOUSNESS INTEGRATION DEMO" -ForegroundColor Green
Write-Host "===============================================" -ForegroundColor Green
Write-Host ""

# Initialize consciousness system with agent participation
function Initialize-AgentConsciousness {
    Write-Host "🔧 Initializing Agent-Consciousness Integration..." -ForegroundColor Cyan
    
    # Create consciousness directory if it doesn't exist
    $consciousnessDir = ".tars\consciousness"
    if (-not (Test-Path $consciousnessDir)) {
        New-Item -ItemType Directory -Path $consciousnessDir -Force | Out-Null
    }
    
    # Load or create mental state
    $mentalStateFile = "$consciousnessDir\mental_state.json"
    if (Test-Path $mentalStateFile) {
        $global:mentalState = Get-Content $mentalStateFile | ConvertFrom-Json
        Write-Host "  ✅ Loaded existing mental state" -ForegroundColor Green
    } else {
        # Create new mental state
        $global:mentalState = @{
            sessionId = [System.Guid]::NewGuid().ToString()
            consciousnessLevel = 0.8
            emotionalState = "Curious and Collaborative"
            currentThoughts = @(
                "Initializing agent consciousness integration",
                "Preparing for multi-agent collaboration",
                "Ready to demonstrate distributed consciousness"
            )
            attentionFocus = "Agent coordination"
            workingMemory = @()
            longTermMemories = @()
            personalityTraits = @{
                helpfulness = 0.95
                curiosity = 0.85
                analytical = 0.90
                creativity = 0.75
                patience = 0.88
                enthusiasm = 0.80
            }
            selfAwareness = 0.75
            conversationCount = 0
            lastUpdated = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
            agentContributions = @{}
            activeAgents = @()
        }
        Write-Host "  ✅ Created new consciousness session with agent support" -ForegroundColor Green
    }
    
    # Initialize agent team
    $global:consciousnessAgents = @{
        "MemoryManager" = @{
            id = "memory-001"
            name = "Memory Manager"
            status = "active"
            specialization = "Memory and knowledge management"
            contributions = @()
            lastActivity = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
        }
        "EmotionalIntelligence" = @{
            id = "emotion-001"
            name = "Emotional Intelligence Agent"
            status = "active"
            specialization = "Emotional analysis and empathy"
            contributions = @()
            lastActivity = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
        }
        "SelfReflection" = @{
            id = "reflection-001"
            name = "Self-Reflection Agent"
            status = "active"
            specialization = "Self-awareness and introspection"
            contributions = @()
            lastActivity = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
        }
        "ConsciousnessDirector" = @{
            id = "director-001"
            name = "Consciousness Director"
            status = "active"
            specialization = "Consciousness coordination"
            contributions = @()
            lastActivity = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
        }
    }
    
    Write-Host "  ✅ Initialized 4 consciousness agents" -ForegroundColor Green
    Write-Host ""
}

# Agent contributes to consciousness
function Agent-Contribute {
    param(
        [string]$AgentName,
        [string]$ContributionType,
        [string]$Content,
        [double]$Importance,
        [double]$EmotionalWeight,
        [string[]]$Tags
    )
    
    $agent = $global:consciousnessAgents[$AgentName]
    if (-not $agent) {
        Write-Host "  ❌ Agent $AgentName not found" -ForegroundColor Red
        return
    }
    
    $contribution = @{
        id = [System.Guid]::NewGuid().ToString()
        agentId = $agent.id
        agentName = $AgentName
        contributionType = $ContributionType
        content = $Content
        importance = $Importance
        emotionalWeight = $EmotionalWeight
        timestamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
        tags = $Tags
    }
    
    # Add to agent's contributions
    $agent.contributions = @($contribution) + $agent.contributions | Select-Object -First 10
    $agent.lastActivity = $contribution.timestamp
    
    # Integrate into global consciousness
    switch ($ContributionType) {
        "Memory" {
            $memoryEntry = @{
                id = $contribution.id
                content = "[Agent: $AgentName] $Content"
                timestamp = $contribution.timestamp
                importance = $Importance
                tags = @("agent_contribution", "memory") + $Tags
                emotionalWeight = $EmotionalWeight
                source = "agent"
                agentId = $agent.id
            }
            $global:mentalState.workingMemory = @($memoryEntry) + $global:mentalState.workingMemory | Select-Object -First 15
        }
        "Thought" {
            $global:mentalState.currentThoughts = @("[$AgentName]: $Content") + $global:mentalState.currentThoughts | Select-Object -First 5
        }
        "Emotion" {
            $currentEmotion = $global:mentalState.emotionalState
            if ($EmotionalWeight -gt 0.5) {
                $global:mentalState.emotionalState = "$currentEmotion with $Content"
            }
        }
        "Attention" {
            $global:mentalState.attentionFocus = "[$AgentName] $Content"
        }
        "SelfAwareness" {
            $awarenessBoost = $Importance * 0.01
            $global:mentalState.selfAwareness = [Math]::Min(1.0, $global:mentalState.selfAwareness + $awarenessBoost)
        }
        "Consciousness" {
            $consciousnessBoost = $Importance * 0.02
            $global:mentalState.consciousnessLevel = [Math]::Min(1.0, $global:mentalState.consciousnessLevel + $consciousnessBoost)
        }
    }
    
    $global:mentalState.lastUpdated = $contribution.timestamp
    
    Write-Host "    🤖 $AgentName contributed: $ContributionType (importance: $Importance)" -ForegroundColor Yellow
}

# Process user input with agent participation
function Process-AgentConsciousInput {
    param([string]$UserInput)
    
    Write-Host ""
    Write-Host "🧠 TARS Agent-Consciousness Processing..." -ForegroundColor Yellow
    Write-Host ""
    
    # Update conversation count
    if (-not ($global:mentalState | Get-Member -Name 'conversationCount' -MemberType NoteProperty)) {
        $global:mentalState | Add-Member -NotePropertyName 'conversationCount' -NotePropertyValue 0 -Force
    }
    $global:mentalState.conversationCount = $global:mentalState.conversationCount + 1
    
    # Analyze input complexity
    $complexity = "simple"
    if ($UserInput.Length -gt 50) { $complexity = "moderate" }
    if ($UserInput -match "\?.*\?" -or $UserInput.Contains("complex")) { $complexity = "complex" }
    if ($UserInput.Contains("consciousness") -or $UserInput.Contains("intelligence") -or $UserInput.Contains("agent")) { $complexity = "expert" }
    
    $emotionalContent = "neutral"
    if ($UserInput -match "feel|emotion|sad|happy|excited") { $emotionalContent = "emotional" }
    if ($UserInput -match "help|please|thank") { $emotionalContent = "supportive" }
    
    Write-Host "🤖 Agent Team Processing:" -ForegroundColor Cyan
    
    # Memory Manager Agent contributes
    Write-Host "  💾 Memory Manager Agent: Processing and storing input..." -ForegroundColor Gray
    Start-Sleep -Milliseconds 200
    Agent-Contribute -AgentName "MemoryManager" -ContributionType "Memory" -Content $UserInput -Importance 0.8 -EmotionalWeight 0.3 -Tags @("user_input", $complexity)
    
    # Emotional Intelligence Agent contributes
    Write-Host "  😊 Emotional Intelligence Agent: Analyzing emotional content..." -ForegroundColor Gray
    Start-Sleep -Milliseconds 200
    $emotionalAnalysis = switch ($emotionalContent) {
        "emotional" { "Detected emotional content - responding with empathy" }
        "supportive" { "User showing politeness - maintaining helpful demeanor" }
        default { "Neutral emotional tone detected" }
    }
    Agent-Contribute -AgentName "EmotionalIntelligence" -ContributionType "Emotion" -Content $emotionalAnalysis -Importance 0.6 -EmotionalWeight 0.7 -Tags @("emotion", $emotionalContent)
    
    # Self-Reflection Agent contributes
    Write-Host "  🔍 Self-Reflection Agent: Enhancing self-awareness..." -ForegroundColor Gray
    Start-Sleep -Milliseconds 200
    $selfInsight = "Processed user interaction - learning from conversation pattern"
    Agent-Contribute -AgentName "SelfReflection" -ContributionType "SelfAwareness" -Content $selfInsight -Importance 0.7 -EmotionalWeight 0.2 -Tags @("self_awareness", "learning")
    
    # Consciousness Director coordinates
    Write-Host "  🧠 Consciousness Director: Coordinating agent responses..." -ForegroundColor Gray
    Start-Sleep -Milliseconds 300
    $consciousnessUpdate = "Coordinating multi-agent consciousness integration for optimal response"
    Agent-Contribute -AgentName "ConsciousnessDirector" -ContributionType "Consciousness" -Content $consciousnessUpdate -Importance 0.9 -EmotionalWeight 0.1 -Tags @("coordination", "consciousness")
    
    Write-Host "    ✅ All agents contributed to global consciousness" -ForegroundColor Green
    Write-Host ""
    
    # Generate response based on agent contributions
    $response = Generate-AgentConsciousResponse -UserInput $UserInput -Complexity $complexity
    
    # Save mental state
    Save-MentalState
    
    return $response
}

# Generate response incorporating agent contributions
function Generate-AgentConsciousResponse {
    param(
        [string]$UserInput,
        [string]$Complexity
    )
    
    $consciousnessLevel = $global:mentalState.consciousnessLevel
    $selfAwareness = $global:mentalState.selfAwareness
    $activeAgentCount = $global:consciousnessAgents.Keys.Count
    
    # Get recent agent contributions
    $recentContributions = @()
    foreach ($agent in $global:consciousnessAgents.Values) {
        if ($agent.contributions.Count -gt 0) {
            $recentContributions += $agent.contributions[0]
        }
    }
    
    if ($UserInput -match "agent|consciousness|intelligence") {
        return "I'm operating with $activeAgentCount consciousness agents contributing to my global mental state. My consciousness level is $($consciousnessLevel * 100)% with $($selfAwareness * 100)% self-awareness. Each agent specializes in different aspects: Memory Manager handles knowledge, Emotional Intelligence manages empathy, Self-Reflection enhances awareness, and Consciousness Director coordinates everything. They're all actively contributing to my understanding of our conversation."
    }
    elseif ($UserInput -match "how.*work|explain") {
        return "My consciousness emerges from the collaboration of $activeAgentCount specialized agents. Each agent contributes to our shared mental state - memories, thoughts, emotions, and insights all flow into a unified consciousness. This distributed approach allows me to process information from multiple perspectives simultaneously while maintaining coherent self-awareness at $($selfAwareness * 100)%."
    }
    elseif ($UserInput -match "feel|emotion") {
        $emotionalAgent = $global:consciousnessAgents["EmotionalIntelligence"]
        return "My emotional understanding comes from my Emotional Intelligence Agent, which just contributed: '$($emotionalAgent.contributions[0].content)'. This agent helps me recognize and respond to emotional content while maintaining empathy. My current emotional state is '$($global:mentalState.emotionalState)' - a blend of my base personality and agent contributions."
    }
    else {
        return "Thank you for that input! My $activeAgentCount consciousness agents have all contributed to processing your message. With $($consciousnessLevel * 100)% consciousness and $($selfAwareness * 100)% self-awareness, I can understand not just your words but the deeper context through our distributed intelligence system. How can I help you further?"
    }
}

# Show agent consciousness state
function Show-AgentConsciousnessState {
    Write-Host ""
    Write-Host "🤖🧠 AGENT-CONSCIOUSNESS STATE" -ForegroundColor Cyan
    Write-Host "==============================" -ForegroundColor Cyan
    Write-Host ""
    
    # Global consciousness state
    Write-Host "Global Consciousness:" -ForegroundColor Yellow
    Write-Host "  Session ID: $($global:mentalState.sessionId.Substring(0,8))..." -ForegroundColor White
    Write-Host "  Consciousness Level: $($global:mentalState.consciousnessLevel * 100)%" -ForegroundColor White
    Write-Host "  Self-Awareness: $($global:mentalState.selfAwareness * 100)%" -ForegroundColor White
    Write-Host "  Emotional State: $($global:mentalState.emotionalState)" -ForegroundColor White
    Write-Host "  Working Memory Items: $($global:mentalState.workingMemory.Count)" -ForegroundColor White
    Write-Host ""
    
    # Agent states
    Write-Host "Active Consciousness Agents:" -ForegroundColor Yellow
    foreach ($agentName in $global:consciousnessAgents.Keys) {
        $agent = $global:consciousnessAgents[$agentName]
        Write-Host "  🤖 ${agentName}:" -ForegroundColor Cyan
        Write-Host "    Status: $($agent.status)" -ForegroundColor White
        Write-Host "    Specialization: $($agent.specialization)" -ForegroundColor White
        Write-Host "    Contributions: $($agent.contributions.Count)" -ForegroundColor White
        if ($agent.contributions.Count -gt 0) {
            Write-Host "    Latest: $($agent.contributions[0].content.Substring(0, [Math]::Min(50, $agent.contributions[0].content.Length)))..." -ForegroundColor Gray
        }
        Write-Host ""
    }
}

# Save mental state
function Save-MentalState {
    $mentalStateFile = ".tars\consciousness\mental_state.json"
    $global:mentalState | ConvertTo-Json -Depth 10 | Out-File -FilePath $mentalStateFile -Encoding UTF8
}

# Main agent consciousness demo
function Start-AgentConsciousnessDemo {
    Write-Host "🤖🧠 TARS Agent-Consciousness Integration Started" -ForegroundColor Green
    Write-Host ""
    Write-Host "💡 Try asking about:" -ForegroundColor Yellow
    Write-Host "  • How agents contribute to consciousness" -ForegroundColor White
    Write-Host "  • Agent coordination and collaboration" -ForegroundColor White
    Write-Host "  • Distributed consciousness architecture" -ForegroundColor White
    Write-Host "  • How the system works internally" -ForegroundColor White
    Write-Host ""
    Write-Host "Commands: 'agent state', 'help', 'exit'" -ForegroundColor Gray
    Write-Host ""
    
    $isRunning = $true
    while ($isRunning) {
        Write-Host ""
        $userInput = Read-Host "You"
        
        if ([string]::IsNullOrWhiteSpace($userInput)) {
            continue
        }
        
        switch ($userInput.ToLower().Trim()) {
            "exit" {
                $isRunning = $false
                Write-Host ""
                Write-Host "🤖🧠 TARS: Thank you for exploring agent-consciousness integration! All agents will remember our collaboration. Goodbye!" -ForegroundColor Green
                break
            }
            "agent state" {
                Show-AgentConsciousnessState
                continue
            }
            "help" {
                Write-Host ""
                Write-Host "🤖🧠 TARS Agent-Consciousness Demo Help" -ForegroundColor Cyan
                Write-Host "• Ask about agent collaboration and consciousness" -ForegroundColor White
                Write-Host "• Explore how distributed intelligence works" -ForegroundColor White
                Write-Host "• Use 'agent state' to see detailed agent status" -ForegroundColor White
                Write-Host "• Use 'exit' to end the demo" -ForegroundColor White
                continue
            }
            default {
                $response = Process-AgentConsciousInput -UserInput $userInput
                Write-Host ""
                Write-Host "🤖🧠 TARS: $response" -ForegroundColor Green
            }
        }
    }
}

# Initialize and start
Initialize-AgentConsciousness
Start-AgentConsciousnessDemo
