# Install-Prerequisites.ps1
# This script installs and configures the prerequisites for TARS

# Function to display colored text
function Write-ColorText {
    param (
        [string]$Text,
        [string]$Color = "White"
    )
    
    Write-Host $Text -ForegroundColor $Color
}

# Function to check if Docker is installed
function Test-DockerInstalled {
    try {
        $dockerVersion = docker --version
        return $true
    }
    catch {
        return $false
    }
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

# Function to update appsettings.json
function Update-AppSettings {
    param (
        [string]$AppSettingsPath
    )
    
    try {
        Write-ColorText "Updating appsettings.json..." "Yellow"
        
        $appSettings = Get-Content $AppSettingsPath -Raw | ConvertFrom-Json
        
        # Update Ollama settings
        $appSettings.Ollama.BaseUrl = "http://localhost:8080"
        $appSettings.Ollama.UseDocker = $true
        
        # Save the updated settings
        $appSettings | ConvertTo-Json -Depth 10 | Set-Content $AppSettingsPath
        
        Write-ColorText "appsettings.json updated successfully" "Green"
    }
    catch {
        Write-ColorText "Error updating appsettings.json: $_" "Red"
    }
}

# Main script
Write-ColorText "TARS Prerequisites Installation" "Cyan"
Write-ColorText "=============================" "Cyan"

# Check if Docker is installed
if (Test-DockerInstalled) {
    Write-ColorText "Docker is installed" "Green"
}
else {
    Write-ColorText "Docker is not installed. Please install Docker Desktop first." "Red"
    Write-ColorText "Download from: https://www.docker.com/products/docker-desktop/" "Yellow"
    exit 1
}

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

# Update appsettings.json
Update-AppSettings "appsettings.json"

# Check if Ollama container is running
$ollamaRunning = docker ps | Select-String "ollama"
if ($ollamaRunning) {
    Write-ColorText "Ollama container is running" "Green"
}
else {
    Write-ColorText "Starting Ollama container..." "Yellow"
    docker run -d --name tars-model-runner -p 8080:11434 --network tars-network ollama/ollama
    Write-ColorText "Ollama container started" "Green"
}

# Start Docker AI Agent
Write-ColorText "Setting up Docker AI Agent..." "Yellow"
$dockerAIAgentExists = Test-Path "docker-compose-docker-ai-agent.yml"
if ($dockerAIAgentExists) {
    Write-ColorText "Starting Docker AI Agent..." "Yellow"
    docker-compose -f docker-compose-docker-ai-agent.yml up -d
    Write-ColorText "Docker AI Agent started" "Green"
}
else {
    Write-ColorText "Docker AI Agent compose file not found. Please run the TARS CLI to set it up." "Yellow"
}

Write-ColorText "Prerequisites installation completed" "Cyan"
Write-ColorText "You can now run TARS CLI with: .\tarscli.cmd" "Cyan"
