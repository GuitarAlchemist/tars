# Test-AutoCodingDocker.ps1
# This script tests auto-coding with Docker

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
Write-ColorText "Auto-Coding with Docker Test" "Cyan"
Write-ColorText "=========================" "Cyan"

# Check if Docker is running
if (Test-DockerRunning) {
    Write-ColorText "Docker is running" "Green"
}
else {
    Write-ColorText "Docker is not running. Please start Docker Desktop first." "Red"
    exit 1
}

# Check if MCP agent is running
$mcpAgentRunning = docker ps | Select-String "tars-mcp-agent"
if ($mcpAgentRunning) {
    Write-ColorText "MCP agent is running" "Green"
}
else {
    Write-ColorText "MCP agent is not running. Please start the MCP agent first." "Red"
    exit 1
}

# Create a test file
$testFilePath = "AutoCodingTest.cs"
$testFileContent = @"
// This is a test file for auto-coding
// The code below is intentionally incomplete and should be improved by TARS

using System;

namespace AutoCodingTest
{
    public class Calculator
    {
        // TODO: Implement Add method
        
        // TODO: Implement Subtract method
        
        // TODO: Implement Multiply method
        
        // TODO: Implement Divide method
    }
}
"@

Write-ColorText "Creating test file: $testFilePath" "Yellow"
Set-Content -Path $testFilePath -Value $testFileContent
Write-ColorText "Test file created" "Green"

# Send the auto-coding request to the MCP agent
Write-ColorText "Sending auto-coding request to MCP agent..." "Yellow"

$requestBody = @{
    command = "code"
    target = @"
Implement the Calculator class with the following methods:
- Add(a, b): Returns the sum of a and b
- Subtract(a, b): Returns the difference of a and b
- Multiply(a, b): Returns the product of a and b
- Divide(a, b): Returns the quotient of a and b

The file is located at $testFilePath
"@
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "http://localhost:8999/api/execute" -Method Post -Body $requestBody -ContentType "application/json"
    
    Write-ColorText "Auto-coding request sent" "Green"
    Write-ColorText "Response: $response" "White"
    
    # Wait for the auto-coding to complete
    Write-ColorText "Waiting for auto-coding to complete..." "Yellow"
    Start-Sleep -Seconds 5
    
    # Check if the file was improved
    $improvedContent = Get-Content -Path $testFilePath -Raw
    
    if ($improvedContent -ne $testFileContent) {
        Write-ColorText "Auto-coding successful!" "Green"
        Write-ColorText "Improved file content:" "Green"
        Write-ColorText $improvedContent "White"
    }
    else {
        Write-ColorText "Auto-coding failed. File was not improved." "Red"
    }
}
catch {
    Write-ColorText "Error sending auto-coding request: $_" "Red"
}

Write-ColorText "Auto-coding test completed" "Cyan"
