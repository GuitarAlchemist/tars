# Test-SwarmImprovements.ps1
# This script tests the improvements made by the swarm

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
Write-ColorText "TARS Swarm Improvements Testing" "Cyan"
Write-ColorText "============================" "Cyan"

# Check if Docker is running
if (Test-DockerRunning) {
    Write-ColorText "Docker is running" "Green"
}
else {
    Write-ColorText "Docker is not running. Please start Docker Desktop first." "Red"
    exit 1
}

# Check if the tester container is running
$testerRunning = docker ps | Select-String "tars-tester"
if (-not $testerRunning) {
    Write-ColorText "Tester container is not running. Please start the swarm first." "Red"
    exit 1
}

# Get the list of improved files
Write-ColorText "Getting list of improved files..." "Yellow"
$response = Invoke-RestMethod -Uri "http://localhost:8990/api/swarm/improvements" -Method Get
$improvedFiles = $response.improvedFiles

if ($improvedFiles.Count -eq 0) {
    Write-ColorText "No improved files found" "Yellow"
    exit 0
}

Write-ColorText "Found $($improvedFiles.Count) improved files" "Green"

# Test each improved file
foreach ($file in $improvedFiles) {
    $filePath = $file.filePath
    $improvementPath = $file.improvementPath
    
    Write-ColorText "Testing file: $filePath" "White"
    
    # Run tests for the file
    $testResponse = Invoke-RestMethod -Uri "http://localhost:8993/api/test" -Method Post -Body (@{
        filePath = $filePath
        improvementPath = $improvementPath
    } | ConvertTo-Json) -ContentType "application/json"
    
    $testsPassed = $testResponse.passed
    $testResults = $testResponse.results
    
    if ($testsPassed) {
        Write-ColorText "  Tests passed" "Green"
    }
    else {
        Write-ColorText "  Tests failed" "Red"
        
        # Show test failures
        foreach ($failure in $testResults.failures) {
            Write-ColorText "    $($failure.testName): $($failure.message)" "Red"
        }
    }
    
    # Update the file status
    Invoke-RestMethod -Uri "http://localhost:8990/api/swarm/update-status" -Method Post -Body (@{
        filePath = $filePath
        testsPassed = $testsPassed
        testResults = $testResults
    } | ConvertTo-Json) -ContentType "application/json" | Out-Null
}

Write-ColorText "Testing completed" "Cyan"
