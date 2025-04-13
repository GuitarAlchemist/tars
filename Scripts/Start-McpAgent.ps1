# Start-McpAgent.ps1
# This script builds and starts the MCP agent

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

# Function to create Docker network
function Create-DockerNetwork {
    param (
        [string]$NetworkName
    )
    
    try {
        $networkExists = docker network ls | Select-String $NetworkName
        
        if (-not $networkExists) {
            Write-ColorText "Creating Docker network: $NetworkName" "Yellow"
            docker network create $NetworkName
            Write-ColorText "Docker network created: $NetworkName" "Green"
        }
        else {
            Write-ColorText "Docker network already exists: $NetworkName" "Green"
        }
    }
    catch {
        Write-ColorText "Error creating Docker network: $_" "Red"
    }
}

# Main script
Write-ColorText "MCP Agent Setup" "Cyan"
Write-ColorText "==============" "Cyan"

# Check if Docker is running
if (Test-DockerRunning) {
    Write-ColorText "Docker is running" "Green"
}
else {
    Write-ColorText "Docker is not running. Please start Docker Desktop first." "Red"
    exit 1
}

# Create Docker network
Create-DockerNetwork "tars-network"

# Build and start the MCP agent
Write-ColorText "Building and starting MCP agent..." "Yellow"
docker-compose -f docker-compose-mcp-agent-simple.yml up -d --build

# Check if the MCP agent is running
$containerRunning = docker ps | Select-String "tars-mcp-agent"
if ($containerRunning) {
    Write-ColorText "MCP agent is running" "Green"
}
else {
    Write-ColorText "Failed to start MCP agent" "Red"
    exit 1
}

Write-ColorText "MCP agent setup completed" "Cyan"
Write-ColorText "MCP agent is running at http://localhost:8999/" "Green"
