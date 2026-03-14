# Run-AutoCoding-Docker.ps1
# This script runs auto-coding using Docker

# Function to display colored text
function Write-ColorText {
    param (
        [string]$Text,
        [string]$Color = "White"
    )
    
    Write-Host $Text -ForegroundColor $Color
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

# Main script
Write-ColorText "TARS Auto-Coding with Docker" "Cyan"
Write-ColorText "=========================" "Cyan"

# Check if Docker is running
if (Test-DockerRunning) {
    Write-ColorText "Docker is running" "Green"
}
else {
    Write-ColorText "Docker is not running. Please start Docker Desktop first." "Red"
    exit 1
}

# Create a Docker network if it doesn't exist
$networkExists = docker network ls | Select-String "tars-network"
if (-not $networkExists) {
    Write-ColorText "Creating Docker network: tars-network" "Yellow"
    docker network create tars-network
    Write-ColorText "Docker network created: tars-network" "Green"
}

# Start the Docker containers
Write-ColorText "Starting Docker containers..." "Yellow"
docker-compose -f docker-compose-auto-coding.yml up -d

# Wait for the containers to start
Write-ColorText "Waiting for containers to start..." "Yellow"
Start-Sleep -Seconds 10

# Pull the model
Write-ColorText "Pulling llama3 model..." "Yellow"
docker exec -it tars-model-runner ollama pull llama3
Write-ColorText "Model pulled successfully" "Green"

# Find the TARS CLI executable
$tarsCli = "TarsCli\bin\Debug\net9.0\tarscli.dll"
if (Test-Path $tarsCli) {
    Write-ColorText "Found TARS CLI at: $tarsCli" "Green"
    
    # Set environment variables for Docker
    $env:OLLAMA_USE_DOCKER = "true"
    $env:OLLAMA_BASE_URL = "http://localhost:8080"
    
    # Run the auto-coding command
    Write-ColorText "Running auto-coding command..." "Yellow"
    dotnet $tarsCli self-code improve TarsCli\Examples\AutoCodingExample.cs --model llama3 --auto-apply
    
    # Check if the file was improved
    $originalContent = @"
using System;
using System.Collections.Generic;

namespace TarsCli.Examples
{
    /// <summary>
    /// Example class for demonstrating auto-coding
    /// </summary>
    public class AutoCodingExample
    {
        // TODO: Implement a method to add two numbers
        
        // TODO: Implement a method to subtract two numbers
        
        // TODO: Implement a method to multiply two numbers
        
        // TODO: Implement a method to divide two numbers
        
        // TODO: Implement a method to calculate the average of a list of numbers
        
        // TODO: Implement a method to find the maximum value in a list of numbers
        
        // TODO: Implement a method to find the minimum value in a list of numbers
    }
}
"@
    
    $improvedContent = Get-Content -Path TarsCli\Examples\AutoCodingExample.cs -Raw
    
    if ($improvedContent -ne $originalContent) {
        Write-ColorText "Auto-coding successful!" "Green"
        Write-ColorText "Improved file content:" "Green"
        Write-ColorText $improvedContent "White"
    }
    else {
        Write-ColorText "Auto-coding failed. File was not improved." "Red"
    }
}
else {
    Write-ColorText "TARS CLI not found at: $tarsCli" "Red"
    Write-ColorText "Please build the solution first" "Red"
    exit 1
}

Write-ColorText "Auto-coding completed" "Cyan"
