#!/usr/bin/env pwsh
#Requires -Version 7.0

<#
.SYNOPSIS
    Demonstrates TARS Auto-Coding capabilities.

.DESCRIPTION
    This script showcases the auto-coding capabilities of TARS (Transformative Autonomous Reasoning System)
    by running a series of auto-coding demos in a structured and visually appealing way.

.EXAMPLE
    .\Demo-AutoCoding.ps1

.NOTES
    Author: TARS Team
    Version: 1.0
    Date: 2025-03-31
#>

# Function to display colored text
function Write-ColorText {
    param (
        [string]$Text,
        [string]$Color = "White"
    )
    
    Write-Host $Text -ForegroundColor $Color
}

# Function to display a section header
function Write-SectionHeader {
    param (
        [string]$Title
    )
    
    Write-Host ""
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
    Write-Host "  $Title" -ForegroundColor Cyan
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
    Write-Host ""
}

# Function to display a subsection header
function Write-SubsectionHeader {
    param (
        [string]$Title
    )
    
    Write-Host ""
    Write-Host "  $Title" -ForegroundColor Yellow
    Write-Host "  " + "─" * $Title.Length -ForegroundColor Yellow
    Write-Host ""
}

# Function to run a command and display its output
function Invoke-TarsCommand {
    param (
        [string]$Command,
        [string]$Description = "",
        [switch]$Wait = $false
    )
    
    if ($Description) {
        Write-Host "  $Description" -ForegroundColor Gray
    }
    
    Write-Host "  > $Command" -ForegroundColor Green
    
    if ($Wait) {
        Write-Host ""
        Write-Host "  Press Enter to continue..." -ForegroundColor Yellow
        Read-Host | Out-Null
    }
    
    Write-Host ""
    
    # Run the command
    Invoke-Expression $Command
    
    Write-Host ""
}

# Function to check if Docker is running
function Test-DockerRunning {
    try {
        $dockerPs = docker ps
        return $true
    }
    catch {
        return $false
    }
}

# Function to check if a Docker network exists
function Test-DockerNetworkExists {
    param (
        [string]$NetworkName
    )
    
    $networkExists = docker network ls | Select-String $NetworkName
    return $null -ne $networkExists
}

# Main script
Clear-Host

# Display the TARS logo
Write-Host @"
  ████████╗ █████╗ ██████╗ ███████╗
  ╚══██╔══╝██╔══██╗██╔══██╗██╔════╝
     ██║   ███████║██████╔╝███████╗
     ██║   ██╔══██║██╔══██╗╚════██║
     ██║   ██║  ██║██║  ██║███████║
     ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝
"@ -ForegroundColor Cyan

Write-Host "  Transformative Autonomous Reasoning System" -ForegroundColor White
Write-Host "  Auto-Coding Demo" -ForegroundColor White
Write-Host ""
Write-Host "  This demo showcases the auto-coding capabilities of TARS." -ForegroundColor Gray
Write-Host "  Press Ctrl+C at any time to exit the demo." -ForegroundColor Gray
Write-Host ""
Write-Host "  Press Enter to begin..." -ForegroundColor Yellow
Read-Host | Out-Null

# Check if Docker is running
Write-SectionHeader "Checking Prerequisites"

if (Test-DockerRunning) {
    Write-ColorText "  Docker is running" "Green"
}
else {
    Write-ColorText "  Docker is not running. Please start Docker Desktop first." "Red"
    exit 1
}

# Check if the Docker network exists
if (Test-DockerNetworkExists "tars-network") {
    Write-ColorText "  Docker network 'tars-network' exists" "Green"
}
else {
    Write-ColorText "  Creating Docker network 'tars-network'..." "Yellow"
    docker network create tars-network
    Write-ColorText "  Docker network 'tars-network' created" "Green"
}

# Section 1: Docker Auto-Coding
Write-SectionHeader "Docker Auto-Coding"

Write-SubsectionHeader "Simple Auto-Coding"
Invoke-TarsCommand -Command ".\Scripts\AutoCoding\Demos\Test-SwarmAutoCode-Simple.ps1" -Description "Running a simple auto-coding demo" -Wait

# Section 2: Swarm Auto-Coding
Write-SectionHeader "Swarm Auto-Coding"

Write-SubsectionHeader "Swarm Auto-Coding Test"
Invoke-TarsCommand -Command ".\Scripts\AutoCoding\Demos\Run-SwarmAutoCode-Test.ps1" -Description "Running a swarm auto-coding test" -Wait

# Section 3: Auto-Coding with TARS CLI
Write-SectionHeader "Auto-Coding with TARS CLI"

Write-SubsectionHeader "Docker Auto-Coding"
Invoke-TarsCommand -Command "dotnet run --project TarsCli/TarsCli.csproj -- auto-code --docker --demo" -Description "Running Docker auto-coding with TARS CLI" -Wait

Write-SubsectionHeader "Swarm Auto-Coding"
Invoke-TarsCommand -Command "dotnet run --project TarsCli/TarsCli.csproj -- auto-code --swarm --demo" -Description "Running Swarm auto-coding with TARS CLI" -Wait

# Conclusion
Write-SectionHeader "Conclusion"

Write-Host "  TARS Auto-Coding Demo Completed" -ForegroundColor Green
Write-Host ""
Write-Host "  You've seen how TARS can auto-code itself using Docker containers and swarm architecture." -ForegroundColor White
Write-Host "  For more information, see the documentation in the docs/AutoCoding directory." -ForegroundColor White
Write-Host ""
Write-Host "  Thank you for exploring TARS Auto-Coding capabilities!" -ForegroundColor Cyan
Write-Host ""
