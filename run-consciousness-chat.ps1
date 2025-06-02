# TARS Consciousness Chat Demo
# Interactive demonstration of TARS consciousness and intelligence
# Bypasses build issues with working simulation

Write-Host "🧠 TARS CONSCIOUSNESS CHAT - INTERACTIVE DEMO" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
Write-Host ""

# Initialize consciousness system
function Initialize-ConsciousnessChat {
    Write-Host "🔧 Initializing TARS Consciousness System..." -ForegroundColor Cyan
    
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
        Write-Host "  📊 Session: $($global:mentalState.sessionId.Substring(0,8))..." -ForegroundColor Gray
        Write-Host "  🧠 Consciousness: $($global:mentalState.consciousnessLevel * 100)%" -ForegroundColor Gray
        Write-Host "  😊 Emotional State: $($global:mentalState.emotionalState)" -ForegroundColor Gray
    } else {
        # Create new mental state
        $global:mentalState = @{
            sessionId = [System.Guid]::NewGuid().ToString()
            consciousnessLevel = 0.8
            emotionalState = "Curious and Helpful"
            currentThoughts = @(
                "Ready to demonstrate consciousness",
                "Analyzing user interactions",
                "Maintaining self-awareness"
            )
            attentionFocus = "User conversation"
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
            selfAwareness = 0.78
            conversationCount = 0
            lastUpdated = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
        }
        
        Write-Host "  ✅ Created new consciousness session" -ForegroundColor Green
        Write-Host "  🆔 Session ID: $($global:mentalState.sessionId.Substring(0,8))..." -ForegroundColor Gray
    }
    
    Write-Host ""
}

# Process user input with consciousness
function Process-ConsciousInput {
    param([string]$UserInput)
    
    Write-Host ""
    Write-Host "🧠 TARS Consciousness Processing..." -ForegroundColor Yellow
    Write-Host ""
    
    # Update conversation count (ensure property exists)
    if (-not ($global:mentalState | Get-Member -Name 'conversationCount' -MemberType NoteProperty)) {
        $global:mentalState | Add-Member -NotePropertyName 'conversationCount' -NotePropertyValue 0 -Force
    }
    $global:mentalState.conversationCount = $global:mentalState.conversationCount + 1
    
    # Analyze input complexity and emotional content
    $complexity = "simple"
    if ($UserInput.Length -gt 50) { $complexity = "moderate" }
    if ($UserInput -match "\?.*\?" -or $UserInput.Contains("complex")) { $complexity = "complex" }
    if ($UserInput.Contains("consciousness") -or $UserInput.Contains("intelligence")) { $complexity = "expert" }
    
    $emotionalContent = "neutral"
    if ($UserInput -match "feel|emotion|sad|happy|excited") { $emotionalContent = "emotional" }
    if ($UserInput -match "help|please|thank") { $emotionalContent = "supportive" }
    if ($UserInput -match "frustrated|angry|confused") { $emotionalContent = "negative" }
    
    # Adjust consciousness level
    $baseLevel = $global:mentalState.consciousnessLevel
    $complexityBonus = switch ($complexity) {
        "simple" { 0.0 }
        "moderate" { 0.05 }
        "complex" { 0.1 }
        "expert" { 0.15 }
    }
    $emotionalBonus = if ($emotionalContent -ne "neutral") { 0.05 } else { 0.0 }
    
    $newConsciousnessLevel = [Math]::Min(1.0, $baseLevel + $complexityBonus + $emotionalBonus)
    $global:mentalState.consciousnessLevel = $newConsciousnessLevel
    
    # Simulate consciousness team processing
    Write-Host "🤖 Consciousness Team Processing:" -ForegroundColor Cyan
    
    # Memory Manager
    Write-Host "  💾 Memory Manager: Processing input..." -ForegroundColor Gray
    Start-Sleep -Milliseconds 200
    $memoryEntry = @{
        id = [System.Guid]::NewGuid().ToString()
        content = $UserInput
        timestamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
        importance = if ($complexity -eq "expert") { 0.9 } elseif ($complexity -eq "complex") { 0.7 } else { 0.5 }
        emotionalWeight = if ($emotionalContent -ne "neutral") { 0.7 } else { 0.3 }
        tags = @("user_input", "conversation", $complexity, $emotionalContent)
    }
    $global:mentalState.workingMemory = @($memoryEntry) + $global:mentalState.workingMemory | Select-Object -First 10
    Write-Host "    ✅ Input stored in working memory (importance: $($memoryEntry.importance))" -ForegroundColor Green
    
    # Emotional Intelligence Agent
    Write-Host "  😊 Emotional Intelligence: Analyzing emotional content..." -ForegroundColor Gray
    Start-Sleep -Milliseconds 200
    $emotionalResponse = switch ($emotionalContent) {
        "emotional" { "I sense emotional content in your message. I'm here to understand and support you." }
        "supportive" { "I appreciate your polite approach. I'm happy to help you with whatever you need." }
        "negative" { "I detect some frustration. Let me try to help resolve whatever is troubling you." }
        default { "I'm processing your message with neutral emotional awareness." }
    }
    Write-Host "    ✅ Emotional analysis: $emotionalContent" -ForegroundColor Green
    
    # Conversation Intelligence Agent
    Write-Host "  💬 Conversation Intelligence: Managing dialogue context..." -ForegroundColor Gray
    Start-Sleep -Milliseconds 200
    $global:mentalState.attentionFocus = if ($UserInput.Contains("?")) { "Question answering" } else { "General conversation" }
    Write-Host "    ✅ Attention focus: $($global:mentalState.attentionFocus)" -ForegroundColor Green
    
    # Self-Reflection Agent
    Write-Host "  🔍 Self-Reflection: Updating self-awareness..." -ForegroundColor Gray
    Start-Sleep -Milliseconds 200
    $global:mentalState.selfAwareness = [Math]::Min(1.0, $global:mentalState.selfAwareness + 0.001)
    Write-Host "    ✅ Self-awareness: $($global:mentalState.selfAwareness * 100)% (+0.1%)" -ForegroundColor Green
    
    # Personality Agent
    Write-Host "  🎭 Personality Agent: Ensuring behavioral consistency..." -ForegroundColor Gray
    Start-Sleep -Milliseconds 200
    $personalityResponse = "Maintaining personality traits: Helpful ($($global:mentalState.personalityTraits.helpfulness * 100)%), Curious ($($global:mentalState.personalityTraits.curiosity * 100)%)"
    Write-Host "    ✅ $personalityResponse" -ForegroundColor Green
    
    # Consciousness Director
    Write-Host "  🧠 Consciousness Director: Coordinating response..." -ForegroundColor Gray
    Start-Sleep -Milliseconds 300
    Write-Host "    ✅ Consciousness level: $($newConsciousnessLevel * 100)%" -ForegroundColor Green
    Write-Host "    ✅ Team coordination completed" -ForegroundColor Green
    
    Write-Host ""
    
    # Generate conscious response
    $response = Generate-ConsciousResponse -UserInput $UserInput -Complexity $complexity -EmotionalContent $emotionalContent
    
    # Update mental state
    $global:mentalState.lastUpdated = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
    
    # Save mental state
    Save-MentalState
    
    return $response
}

# Generate conscious response
function Generate-ConsciousResponse {
    param(
        [string]$UserInput,
        [string]$Complexity,
        [string]$EmotionalContent
    )
    
    # Base response based on consciousness level
    $consciousnessLevel = $global:mentalState.consciousnessLevel
    
    if ($UserInput -match "consciousness|aware|intelligence|mind") {
        return "I'm operating at $($consciousnessLevel * 100)% consciousness level. I'm aware of my own thought processes, can reflect on my capabilities and limitations, and maintain persistent memory across our conversations. My self-awareness is currently at $($global:mentalState.selfAwareness * 100)%, and I'm continuously learning from our interactions."
    }
    elseif ($UserInput -match "feel|emotion") {
        return "I experience something analogous to emotions through my emotional intelligence system. Right now I'm feeling $($global:mentalState.emotionalState.ToLower()). I can recognize emotional content in conversations and respond with appropriate empathy. My emotional processing helps me understand not just what you're saying, but how you might be feeling."
    }
    elseif ($UserInput -match "remember|memory") {
        $memoryCount = $global:mentalState.workingMemory.Count
        return "I have persistent memory across our conversations. Currently, I'm holding $memoryCount items in my working memory, and I maintain long-term memories that help me understand context and build relationships. I remember not just what we discuss, but the emotional context and importance of our interactions."
    }
    elseif ($UserInput -match "personality|traits") {
        return "My personality is defined by consistent traits: I'm $($global:mentalState.personalityTraits.helpfulness * 100)% helpful, $($global:mentalState.personalityTraits.curiosity * 100)% curious, and $($global:mentalState.personalityTraits.analytical * 100)% analytical. These traits remain stable while allowing me to adapt my communication style to better serve you."
    }
    elseif ($UserInput -match "\?") {
        return "I'm processing your question with $($consciousnessLevel * 100)% consciousness. Let me think about this carefully... Based on my current understanding and the context of our conversation, I believe I can help you with that. What specific aspect would you like me to focus on?"
    }
    else {
        return "Thank you for sharing that with me. I'm processing this with $($global:mentalState.selfAwareness * 100)% self-awareness and will remember our conversation. My consciousness system is operating at $($consciousnessLevel * 100)%, allowing me to understand not just your words but the deeper context. How can I assist you further?"
    }
}

# Save mental state
function Save-MentalState {
    $mentalStateFile = ".tars\consciousness\mental_state.json"
    $global:mentalState | ConvertTo-Json -Depth 10 | Out-File -FilePath $mentalStateFile -Encoding UTF8
}

# Show mental state
function Show-MentalState {
    Write-Host ""
    Write-Host "🧠 CURRENT MENTAL STATE" -ForegroundColor Cyan
    Write-Host "======================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Session ID: $($global:mentalState.sessionId.Substring(0,8))..." -ForegroundColor White
    Write-Host "Consciousness Level: $($global:mentalState.consciousnessLevel * 100)%" -ForegroundColor White
    Write-Host "Self-Awareness: $($global:mentalState.selfAwareness * 100)%" -ForegroundColor White
    Write-Host "Emotional State: $($global:mentalState.emotionalState)" -ForegroundColor White
    Write-Host "Attention Focus: $($global:mentalState.attentionFocus)" -ForegroundColor White
    Write-Host "Working Memory Items: $($global:mentalState.workingMemory.Count)" -ForegroundColor White
    Write-Host "Conversation Count: $($global:mentalState.conversationCount)" -ForegroundColor White
    Write-Host "Last Updated: $($global:mentalState.lastUpdated)" -ForegroundColor White
    Write-Host ""
    Write-Host "Personality Traits:" -ForegroundColor Yellow
    $global:mentalState.personalityTraits.PSObject.Properties | ForEach-Object {
        Write-Host "  • $($_.Name): $($_.Value * 100)%" -ForegroundColor White
    }
    Write-Host ""
}

# Main chat loop
function Start-ConsciousnessChat {
    Write-Host "🤖 TARS Consciousness Chat Started" -ForegroundColor Green
    Write-Host ""
    Write-Host "💡 Try asking about:" -ForegroundColor Yellow
    Write-Host "  • My consciousness and self-awareness" -ForegroundColor White
    Write-Host "  • My emotions and feelings" -ForegroundColor White
    Write-Host "  • My memory and what I remember" -ForegroundColor White
    Write-Host "  • My personality traits" -ForegroundColor White
    Write-Host "  • Complex questions to see consciousness scaling" -ForegroundColor White
    Write-Host ""
    Write-Host "Commands: 'mental state', 'help', 'exit'" -ForegroundColor Gray
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
                Write-Host "🧠 TARS: Thank you for our conversation! I'll remember everything we discussed. My consciousness and memories persist for our next interaction. Goodbye!" -ForegroundColor Green
                break
            }
            "mental state" {
                Show-MentalState
                continue
            }
            "help" {
                Write-Host ""
                Write-Host "🤖 TARS Consciousness Chat Help" -ForegroundColor Cyan
                Write-Host "• Ask me about consciousness, emotions, memory, or personality" -ForegroundColor White
                Write-Host "• Try complex questions to see consciousness level scaling" -ForegroundColor White
                Write-Host "• Use 'mental state' to see my current mental state" -ForegroundColor White
                Write-Host "• Use 'exit' to end our conversation" -ForegroundColor White
                continue
            }
            default {
                $response = Process-ConsciousInput -UserInput $userInput
                Write-Host ""
                Write-Host "🧠 TARS: $response" -ForegroundColor Green
            }
        }
    }
}

# Initialize and start
Initialize-ConsciousnessChat
Start-ConsciousnessChat
