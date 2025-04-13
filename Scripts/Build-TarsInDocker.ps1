# Build-TarsInDocker.ps1
# This script builds the TARS solution in Docker

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
Write-ColorText "TARS Docker Build" "Cyan"
Write-ColorText "================" "Cyan"

# Check if Docker is running
if (Test-DockerRunning) {
    Write-ColorText "Docker is running" "Green"
}
else {
    Write-ColorText "Docker is not running. Please start Docker Desktop first." "Red"
    exit 1
}

# Create a Docker container for building
Write-ColorText "Creating Docker container for building..." "Yellow"
docker run -d --name tars-builder --network tars-network -v ${PWD}:/app/workspace mcr.microsoft.com/dotnet/sdk:9.0
Write-ColorText "Docker container created" "Green"

# Build the solution
Write-ColorText "Building TARS solution..." "Yellow"
$buildOutput = docker exec -it tars-builder bash -c "cd /app/workspace && dotnet build"

# Check if the build succeeded
if ($LASTEXITCODE -eq 0) {
    Write-ColorText "Build succeeded" "Green"
}
else {
    Write-ColorText "Build failed" "Red"
    Write-ColorText $buildOutput "Red"
    
    # Clean up
    Write-ColorText "Cleaning up..." "Yellow"
    docker stop tars-builder
    docker rm tars-builder
    Write-ColorText "Cleanup completed" "Green"
    
    exit 1
}

# Run tests
Write-ColorText "Running tests..." "Yellow"
$testOutput = docker exec -it tars-builder bash -c "cd /app/workspace && dotnet test"

# Check if the tests succeeded
if ($LASTEXITCODE -eq 0) {
    Write-ColorText "Tests succeeded" "Green"
}
else {
    Write-ColorText "Tests failed" "Red"
    Write-ColorText $testOutput "Red"
    
    # Clean up
    Write-ColorText "Cleaning up..." "Yellow"
    docker stop tars-builder
    docker rm tars-builder
    Write-ColorText "Cleanup completed" "Green"
    
    exit 1
}

# Clean up
Write-ColorText "Cleaning up..." "Yellow"
docker stop tars-builder
docker rm tars-builder
Write-ColorText "Cleanup completed" "Green"

Write-ColorText "Build and test completed successfully" "Cyan"
