#!/usr/bin/env powershell
# PowerShell script for Docker AI Agent demo

<#
.SYNOPSIS
    Run Docker AI Agent with TARS integration.

.DESCRIPTION
    This script starts the Docker AI Agent and bridges it with TARS MCP.
    It provides a convenient way to use Docker AI Agent capabilities with TARS.

.EXAMPLE
    .\Run-DockerAIAgent.ps1

.NOTES
    Author: TARS Team
    Version: 1.0
    Date: 2025-04-12
#>

# Configuration
$TarsCliPath = Join-Path $PSScriptRoot "..\TarsCli\bin\Debug\net9.0\tarscli.exe"
$DockerComposeFile = Join-Path $PSScriptRoot "..\docker-compose-docker-ai-agent.yml"
$McpUrl = "http://localhost:8999/"
$DemoDelay = 1.5  # Delay between commands in seconds
$PauseBetweenSections = 3  # Pause between major sections in seconds

# Save current directory and change to repository root
$CurrentDir = Get-Location
Set-Location (Join-Path $PSScriptRoot "..")

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

# Function to invoke a TARS command
function Invoke-TarsCommand {
    param([string]$Command)

    Write-Host "> tarscli $Command" -ForegroundColor Green
    Start-Sleep -Seconds 0.5
    & $TarsCliPath $Command.Split()
    Write-Host ""
}

# Function to pause between demo sections
function Pause-Demo {
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

# Check if TARS CLI exists
if (-not (Test-Path $TarsCliPath)) {
    Write-Host "Error: TARS CLI not found at $TarsCliPath" -ForegroundColor Red
    Write-Host "Please build the TARS CLI project first by running 'dotnet build' in the TarsCli directory." -ForegroundColor Red
    Set-Location $CurrentDir
    exit 1
}

# Introduction
Write-Header "DOCKER AI AGENT WITH TARS INTEGRATION"

Write-Host "This script will demonstrate the Docker AI Agent integration with TARS."
Write-Host "The Docker AI Agent provides access to local LLMs and Docker capabilities."
Write-Host "Press Ctrl+C at any time to exit the demo."
Write-Host ""

Pause-Demo -Message "Starting demo in"

# Section 1: Start Docker AI Agent
Write-Header "STARTING DOCKER AI AGENT"

Write-SubHeader "Starting Docker AI Agent"
Invoke-TarsCommand "docker-ai-agent start"

Pause-Demo

# Section 2: Check Docker AI Agent Status
Write-Header "DOCKER AI AGENT STATUS"

Write-SubHeader "Checking Docker AI Agent Status"
Invoke-TarsCommand "docker-ai-agent status"

Pause-Demo

# Section 3: List Available Models
Write-Header "AVAILABLE MODELS"

Write-SubHeader "Listing Available Models"
Invoke-TarsCommand "docker-ai-agent list-models"

Pause-Demo

# Section 4: Generate Text
Write-Header "TEXT GENERATION"

Write-SubHeader "Generating Text with Docker AI Agent"
Invoke-TarsCommand "docker-ai-agent generate `"Write a short poem about artificial intelligence and creativity`""

Pause-Demo

# Section 5: Execute Shell Command
Write-Header "SHELL COMMAND EXECUTION"

Write-SubHeader "Executing Shell Command with Docker AI Agent"
Invoke-TarsCommand "docker-ai-agent shell `"docker ps`""

Pause-Demo

# Section 6: Bridge with MCP
Write-Header "MCP BRIDGE"

Write-SubHeader "Bridging Docker AI Agent with MCP"
Invoke-TarsCommand "docker-ai-agent bridge --mcp-url $McpUrl"

Pause-Demo

# Section 7: Stop Docker AI Agent
Write-Header "STOPPING DOCKER AI AGENT"

Write-SubHeader "Stopping Docker AI Agent"
Invoke-TarsCommand "docker-ai-agent stop"

# Conclusion
Write-Header "DEMO COMPLETE"

Write-Host "This concludes the demonstration of Docker AI Agent integration with TARS."
Write-Host "The Docker AI Agent provides access to local LLMs and Docker capabilities,"
Write-Host "enabling TARS to leverage Docker's AI features for autonomous self-improvement."
Write-Host ""
Write-Host "For more information, see the Docker AI Agent documentation:"
Write-Host "docs/Docker-AI-Agent.md"
Write-Host ""

# Restore original directory
Set-Location $CurrentDir
