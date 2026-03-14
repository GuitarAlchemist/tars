# Test-AutoCoding.ps1
# This script tests the auto-coding capabilities of TARS

# Function to display colored text
function Write-ColorText {
    param (
        [string]$Text,
        [string]$Color = "White"
    )
    
    Write-Host $Text -ForegroundColor $Color
}

# Main script
Write-ColorText "Testing Auto-Coding Capabilities" "Cyan"
Write-ColorText "=============================" "Cyan"

# Check if Docker is running
Write-ColorText "Checking Docker status..." "Yellow"
try {
    $dockerPs = docker ps
    Write-ColorText "Docker is running" "Green"
}
catch {
    Write-ColorText "Docker is not running. Please start Docker Desktop first." "Red"
    exit 1
}

# Check if Ollama is running
Write-ColorText "Checking Ollama status..." "Yellow"
$ollamaRunning = docker ps | Select-String "ollama"
if ($ollamaRunning) {
    Write-ColorText "Ollama is running" "Green"
}
else {
    Write-ColorText "Ollama is not running" "Red"
    exit 1
}

# Create a simple test file
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

# Run the auto-coding command
Write-ColorText "Running auto-coding command..." "Yellow"
$tarsCli = "TarsCli\bin\Debug\net9.0\tarscli.exe"

if (Test-Path $tarsCli) {
    Write-ColorText "Found TARS CLI at: $tarsCli" "Green"
    
    # Run the auto-coding command
    & $tarsCli self-coding improve --file $testFilePath --model llama3 --auto-apply
    
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
else {
    Write-ColorText "TARS CLI not found at: $tarsCli" "Red"
    Write-ColorText "Please build the solution first" "Red"
    exit 1
}

Write-ColorText "Auto-coding test completed" "Cyan"
