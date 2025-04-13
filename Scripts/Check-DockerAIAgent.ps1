# Check-DockerAIAgent.ps1
# This script checks the status of the Docker AI Agent

# Check if Docker is running
Write-Host "Checking Docker status..." -ForegroundColor Cyan
try {
    $dockerPs = docker ps
    Write-Host "Docker is running" -ForegroundColor Green
}
catch {
    Write-Host "Docker is not running. Please start Docker Desktop first." -ForegroundColor Red
    exit 1
}

# Check if Docker AI Agent is running
Write-Host "Checking Docker AI Agent status..." -ForegroundColor Cyan
$containerRunning = docker ps | Select-String "tars-docker-ai-agent"
if ($containerRunning) {
    Write-Host "Docker AI Agent is running" -ForegroundColor Green
}
else {
    Write-Host "Docker AI Agent is not running" -ForegroundColor Red
    exit 1
}

# Check if Docker AI Agent is accessible
Write-Host "Checking Docker AI Agent accessibility..." -ForegroundColor Cyan
try {
    $response = Invoke-RestMethod -Uri "http://localhost:8997/api/health" -Method Get -ErrorAction SilentlyContinue
    Write-Host "Docker AI Agent is accessible" -ForegroundColor Green
}
catch {
    Write-Host "Docker AI Agent is not accessible" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
}

# Check if MCP is running
Write-Host "Checking MCP status..." -ForegroundColor Cyan
try {
    $response = Invoke-RestMethod -Uri "http://localhost:8999/health" -Method Get -ErrorAction SilentlyContinue
    Write-Host "MCP is accessible" -ForegroundColor Green
}
catch {
    Write-Host "MCP is not accessible" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
}

# Check Docker AI Agent logs
Write-Host "Docker AI Agent logs:" -ForegroundColor Cyan
docker logs tars-docker-ai-agent --tail 20

Write-Host "Docker AI Agent check completed" -ForegroundColor Cyan
