# Start-DockerAIAgent.ps1
# This script starts the Docker AI Agent and bridges it with MCP

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

# Function to start the Docker AI Agent
function Start-DockerAIAgent {
    try {
        Write-ColorText "Starting Docker AI Agent..." "Yellow"
        
        # Check if the Docker AI Agent is already running
        $containerRunning = docker ps | Select-String "tars-docker-ai-agent"
        
        if ($containerRunning) {
            Write-ColorText "Docker AI Agent is already running" "Green"
            return $true
        }
        
        # Start the Docker AI Agent
        docker-compose -f docker-compose-docker-ai-agent.yml up -d
        
        # Check if the Docker AI Agent started successfully
        $containerRunning = docker ps | Select-String "tars-docker-ai-agent"
        
        if ($containerRunning) {
            Write-ColorText "Docker AI Agent started successfully" "Green"
            return $true
        }
        else {
            Write-ColorText "Failed to start Docker AI Agent" "Red"
            return $false
        }
    }
    catch {
        Write-ColorText "Error starting Docker AI Agent: $_" "Red"
        return $false
    }
}

# Function to bridge Docker AI Agent with MCP
function Bridge-DockerAIAgentWithMcp {
    param (
        [string]$McpUrl
    )
    
    try {
        Write-ColorText "Bridging Docker AI Agent with MCP at $McpUrl..." "Yellow"
        
        # Make a POST request to the Docker AI Agent to bridge with MCP
        $requestData = @{
            mcpUrl = $McpUrl
        } | ConvertTo-Json
        
        $response = Invoke-RestMethod -Uri "http://localhost:8997/api/bridge/mcp" -Method Post -Body $requestData -ContentType "application/json"
        
        if ($response.success) {
            Write-ColorText "Docker AI Agent bridged with MCP successfully" "Green"
            return $true
        }
        else {
            Write-ColorText "Failed to bridge Docker AI Agent with MCP: $($response.error)" "Red"
            return $false
        }
    }
    catch {
        Write-ColorText "Error bridging Docker AI Agent with MCP: $_" "Red"
        return $false
    }
}

# Main script
Write-ColorText "Docker AI Agent Setup" "Cyan"
Write-ColorText "===================" "Cyan"

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

# Start Docker AI Agent
if (Start-DockerAIAgent) {
    Write-ColorText "Docker AI Agent is running" "Green"
}
else {
    Write-ColorText "Failed to start Docker AI Agent" "Red"
    exit 1
}

# Bridge Docker AI Agent with MCP
$mcpUrl = "http://localhost:8999/"
if (Bridge-DockerAIAgentWithMcp -McpUrl $mcpUrl) {
    Write-ColorText "Docker AI Agent is bridged with MCP" "Green"
}
else {
    Write-ColorText "Failed to bridge Docker AI Agent with MCP" "Red"
    exit 1
}

Write-ColorText "Docker AI Agent setup completed" "Cyan"
Write-ColorText "You can now use TARS CLI with Docker AI Agent" "Cyan"
