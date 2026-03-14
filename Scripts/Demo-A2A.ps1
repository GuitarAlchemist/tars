#!/usr/bin/env pwsh
#Requires -Version 7.0

<#
.SYNOPSIS
    Demonstrates TARS A2A (Agent-to-Agent) protocol capabilities.

.DESCRIPTION
    This script showcases the A2A protocol implementation in TARS by running
    a series of A2A-related commands and demonstrating agent-to-agent communication.

.EXAMPLE
    .\Demo-A2A.ps1

.NOTES
    Author: TARS Team
    Version: 1.0
    Date: 2025-04-12
#>

# Configuration
$TarsCliPath = Join-Path $PSScriptRoot "..\TarsCli\bin\Debug\net9.0\tarscli.exe"
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
Write-Header "TARS A2A PROTOCOL DEMONSTRATION"

Write-Host "This script will demonstrate the A2A (Agent-to-Agent) protocol capabilities of TARS."
Write-Host "The A2A protocol enables interoperability between different AI agents."
Write-Host "Press Ctrl+C at any time to exit the demo."
Write-Host ""

Pause-Demo -Message "Starting demo in"

# Section 1: A2A Server
Write-Header "A2A SERVER"

Write-SubHeader "Starting the A2A Server"
Invoke-TarsCommand "a2a start"

Pause-Demo

# Section 2: Agent Card
Write-Header "AGENT CARD"

Write-SubHeader "Getting the TARS Agent Card"
Invoke-TarsCommand "a2a get-agent-card --agent-url http://localhost:8998/"

Pause-Demo

# Section 3: Code Generation
Write-Header "CODE GENERATION SKILL"

Write-SubHeader "Sending a Code Generation Task"
Invoke-TarsCommand "a2a send --agent-url http://localhost:8998/ --message `"Generate a C# class for a Customer entity with properties for ID, Name, Email, and Address`" --skill-id code_generation"

Pause-Demo

# Section 4: Code Analysis
Write-Header "CODE ANALYSIS SKILL"

Write-SubHeader "Sending a Code Analysis Task"
$CodeToAnalyze = "public void ProcessData(string data) { var result = data.Split(','); Console.WriteLine(result[0]); }"
Invoke-TarsCommand "a2a send --agent-url http://localhost:8998/ --message `"Analyze this code for potential issues: $CodeToAnalyze`" --skill-id code_analysis"

Pause-Demo

# Section 5: Knowledge Extraction
Write-Header "KNOWLEDGE EXTRACTION SKILL"

Write-SubHeader "Sending a Knowledge Extraction Task"
Invoke-TarsCommand "a2a send --agent-url http://localhost:8998/ --message `"Extract key concepts from the A2A protocol documentation`" --skill-id knowledge_extraction"

Pause-Demo

# Section 6: Self Improvement
Write-Header "SELF IMPROVEMENT SKILL"

Write-SubHeader "Sending a Self Improvement Task"
Invoke-TarsCommand "a2a send --agent-url http://localhost:8998/ --message `"Suggest improvements for the A2A protocol implementation`" --skill-id self_improvement"

Pause-Demo

# Section 7: MCP Bridge
Write-Header "MCP BRIDGE"

Write-SubHeader "Using A2A through MCP"
Invoke-TarsCommand "mcp execute --action a2a --operation send_task --agent_url http://localhost:8998/ --content `"Generate a simple logging class in C#`" --skill_id code_generation"

Pause-Demo

# Section 8: Stopping the Server
Write-Header "STOPPING THE SERVER"

Write-SubHeader "Stopping the A2A Server"
Invoke-TarsCommand "a2a stop"

# Conclusion
Write-Header "DEMO COMPLETE"

Write-Host "This concludes the demonstration of TARS A2A protocol capabilities."
Write-Host "The A2A protocol enables TARS to communicate with other A2A-compatible agents"
Write-Host "and expose its capabilities through a standardized interface."
Write-Host ""
Write-Host "For more information, see the A2A protocol documentation:"
Write-Host "docs/A2A-Protocol.md"
Write-Host ""

# Restore original directory
Set-Location $CurrentDir
