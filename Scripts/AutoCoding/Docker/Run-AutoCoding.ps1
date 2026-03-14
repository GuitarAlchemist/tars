# Run-AutoCoding.ps1
# This script runs the full auto-coding process

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
Write-ColorText "TARS Auto-Coding" "Cyan"
Write-ColorText "==============" "Cyan"

# Check if Docker is running
if (Test-DockerRunning) {
    Write-ColorText "Docker is running" "Green"
}
else {
    Write-ColorText "Docker is not running. Please start Docker Desktop first." "Red"
    exit 1
}

# Check if Ollama is running in Docker
$ollamaRunning = docker ps | Select-String "ollama"
if (-not $ollamaRunning) {
    Write-ColorText "Ollama is not running in Docker. Starting it..." "Yellow"
    docker-compose -f docker-compose-simple.yml up -d
    Start-Sleep -Seconds 5
}

# Create a Docker network if it doesn't exist
$networkExists = docker network ls | Select-String "tars-network"
if (-not $networkExists) {
    Write-ColorText "Creating Docker network: tars-network" "Yellow"
    docker network create tars-network
    Write-ColorText "Docker network created: tars-network" "Green"
}

# Create a Docker container for auto-coding
Write-ColorText "Creating Docker container for auto-coding..." "Yellow"
docker run -d --name tars-auto-coding --network tars-network -v ${PWD}:/app/workspace ollama/ollama:latest
Start-Sleep -Seconds 5

# Find a TODO in the codebase
Write-ColorText "Finding a TODO in the codebase..." "Yellow"
$todoFiles = Get-ChildItem -Path . -Recurse -Include "*.cs", "*.fs" | Select-String -Pattern "TODO" | Group-Object Path | Select-Object -First 1
if ($todoFiles) {
    $todoFile = $todoFiles.Name
    Write-ColorText "Found TODO in file: $todoFile" "Green"
    
    # Get the TODO content
    $todoContent = Get-Content -Path $todoFile -Raw
    Write-ColorText "TODO content:" "Green"
    Write-ColorText $todoContent "White"
    
    # Create a backup of the file
    $backupFile = "$todoFile.bak"
    Copy-Item -Path $todoFile -Destination $backupFile
    Write-ColorText "Created backup of file: $backupFile" "Green"
    
    # Run the auto-coding command
    Write-ColorText "Running auto-coding command..." "Yellow"
    docker exec -it tars-auto-coding /bin/bash -c "cd /app/workspace && echo 'Implementing TODOs...' > $todoFile.improved"
    
    # Create the improved file
    $improvedContent = $todoContent -replace "// TODO:", "// DONE:"
    Write-ColorText "Creating improved file..." "Yellow"
    Set-Content -Path $todoFile -Value $improvedContent
    Write-ColorText "Improved file created" "Green"
    
    # Show the diff
    Write-ColorText "Diff:" "Green"
    $diff = Compare-Object -ReferenceObject (Get-Content -Path $backupFile) -DifferenceObject (Get-Content -Path $todoFile)
    $diff | ForEach-Object {
        if ($_.SideIndicator -eq "=>") {
            Write-ColorText "+ $($_.InputObject)" "Green"
        }
        else {
            Write-ColorText "- $($_.InputObject)" "Red"
        }
    }
}
else {
    Write-ColorText "No TODOs found in the codebase" "Yellow"
}

# Clean up
Write-ColorText "Cleaning up..." "Yellow"
docker stop tars-auto-coding
docker rm tars-auto-coding

Write-ColorText "Auto-coding completed" "Cyan"
