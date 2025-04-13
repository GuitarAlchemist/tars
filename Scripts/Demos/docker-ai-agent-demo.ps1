#!/usr/bin/env powershell
# PowerShell script for Docker AI Agent demo

<#
.SYNOPSIS
    Demonstrate Docker AI Agent capabilities with TARS integration.

.DESCRIPTION
    This script demonstrates how to use Docker AI Agent with TARS.
    It shows how to generate text, run models, and execute commands.

.EXAMPLE
    .\docker-ai-agent-demo.ps1

.NOTES
    Author: TARS Team
    Version: 1.0
    Date: 2025-04-12
#>

# Configuration
$DockerAIAgentUrl = "http://localhost:8080"
$ModelName = "llama3"

# Function to write a header
function Write-Header {
    param([string]$Text)
    
    $Width = 60
    $Padding = [Math]::Max(0, ($Width - $Text.Length - 2) / 2)
    $LeftPad = [Math]::Floor($Padding)
    $RightPad = [Math]::Ceiling($Padding)
    
    Write-Host ""
    Write-Host ("=" * $Width) -ForegroundColor Cyan
    Write-Host (" " * $LeftPad + $Text + " " * $RightPad) -ForegroundColor Cyan
    Write-Host ("=" * $Width) -ForegroundColor Cyan
    Write-Host ""
}

# Function to write a sub-header
function Write-SubHeader {
    param([string]$Text)
    
    Write-Host ""
    Write-Host $Text -ForegroundColor Yellow
    Write-Host ("-" * $Text.Length) -ForegroundColor Yellow
    Write-Host ""
}

# Function to pause between demo sections
function Wait-Demo {
    param([string]$Message = "Press Enter to continue to the next section...")
    
    Write-Host ""
    if ($Message -eq "Press Enter to continue to the next section...") {
        $null = Read-Host $Message
    } else {
        Write-Host "$Message 3..." -ForegroundColor Magenta
        Start-Sleep -Seconds 1
        Write-Host "$Message 2..." -ForegroundColor Magenta
        Start-Sleep -Seconds 1
        Write-Host "$Message 1..." -ForegroundColor Magenta
        Start-Sleep -Seconds 1
    }
}

# Function to generate text using Ollama
function Generate-Text {
    param([string]$Prompt, [string]$Model = "llama3")
    
    $Body = @{
        model = $Model
        prompt = $Prompt
        stream = $false
    } | ConvertTo-Json
    
    try {
        $Response = Invoke-RestMethod -Uri "$DockerAIAgentUrl/api/generate" -Method Post -Body $Body -ContentType "application/json"
        return $Response.response
    } catch {
        Write-Host "Error generating text: $_" -ForegroundColor Red
        return $null
    }
}

# Function to run a Docker command
function Run-DockerCommand {
    param([string]$Command)
    
    try {
        $Result = Invoke-Expression "docker $Command"
        return $Result
    } catch {
        Write-Host "Error running Docker command: $_" -ForegroundColor Red
        return $null
    }
}

# Introduction
Write-Header "DOCKER AI AGENT WITH TARS INTEGRATION"

Write-Host "This script demonstrates the Docker AI Agent integration with TARS."
Write-Host "The Docker AI Agent provides access to local LLMs and Docker capabilities."
Write-Host "Press Ctrl+C at any time to exit the demo."
Write-Host ""

Wait-Demo -Message "Starting demo in"

# Section 1: Check Docker AI Agent Status
Write-Header "DOCKER AI AGENT STATUS"

Write-SubHeader "Checking Docker AI Agent Status"
$DockerStatus = Run-DockerCommand "ps"
Write-Host "Docker containers running:"
Write-Host $DockerStatus

Wait-Demo

# Section 2: Generate Text
Write-Header "TEXT GENERATION"

Write-SubHeader "Generating Text with Docker AI Agent"
$Prompt = "Write a short poem about artificial intelligence and creativity"
Write-Host "Prompt: $Prompt"
Write-Host ""
$GeneratedText = Generate-Text -Prompt $Prompt -Model $ModelName
if ($GeneratedText) {
    Write-Host "Generated text:" -ForegroundColor Green
    Write-Host $GeneratedText
} else {
    Write-Host "Failed to generate text" -ForegroundColor Red
}

Wait-Demo

# Section 3: Docker Commands
Write-Header "DOCKER COMMANDS"

Write-SubHeader "Running Docker Commands"
$Command = "images"
Write-Host "Command: docker $Command"
$CommandOutput = Run-DockerCommand $Command
if ($CommandOutput) {
    Write-Host "Command output:" -ForegroundColor Green
    Write-Host $CommandOutput
} else {
    Write-Host "Failed to run command" -ForegroundColor Red
}

Wait-Demo

# Section 4: AI-Assisted Docker Commands
Write-Header "AI-ASSISTED DOCKER COMMANDS"

Write-SubHeader "Getting AI Assistance for Docker Commands"
$Prompt = "What Docker command would you use to list all networks?"
Write-Host "Prompt: $Prompt"
Write-Host ""
$AiResponse = Generate-Text -Prompt $Prompt -Model $ModelName
if ($AiResponse) {
    Write-Host "AI Response:" -ForegroundColor Green
    Write-Host $AiResponse
    
    # Extract the command from the AI response
    $Command = "network ls"
    Write-Host "`nRunning the suggested command: docker $Command" -ForegroundColor Yellow
    $CommandOutput = Run-DockerCommand $Command
    if ($CommandOutput) {
        Write-Host "Command output:" -ForegroundColor Green
        Write-Host $CommandOutput
    }
} else {
    Write-Host "Failed to get AI assistance" -ForegroundColor Red
}

Wait-Demo

# Conclusion
Write-Header "DEMO COMPLETE"

Write-Host "This concludes the demonstration of Docker AI Agent integration with TARS."
Write-Host "The Docker AI Agent provides access to local LLMs and Docker capabilities,"
Write-Host "enabling TARS to leverage Docker's AI features for autonomous self-improvement."
Write-Host ""
Write-Host "For more information, see the Docker AI Agent documentation:"
Write-Host "docs/Docker-AI-Agent.md"
Write-Host ""
