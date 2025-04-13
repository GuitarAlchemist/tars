# Run-TarsAutoCodeSwarm.ps1
# This script orchestrates the entire auto-coding process using Docker swarm

param (
    [Parameter(Mandatory=$false)]
    [string[]]$TargetDirectories = @("TarsCli", "TarsEngine"),
    
    [Parameter(Mandatory=$false)]
    [int]$AgentCount = 3,
    
    [Parameter(Mandatory=$false)]
    [string]$Model = "llama3",
    
    [Parameter(Mandatory=$false)]
    [switch]$AutoApply = $false,
    
    [Parameter(Mandatory=$false)]
    [switch]$SkipTests = $false
)

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
Write-ColorText "TARS Auto-Code Swarm" "Cyan"
Write-ColorText "===================" "Cyan"

# Display parameters
Write-ColorText "Parameters:" "Yellow"
Write-ColorText "  Target Directories: $($TargetDirectories -join ', ')" "Yellow"
Write-ColorText "  Agent Count: $AgentCount" "Yellow"
Write-ColorText "  Model: $Model" "Yellow"
Write-ColorText "  Auto Apply: $AutoApply" "Yellow"
Write-ColorText "  Skip Tests: $SkipTests" "Yellow"

# Check if Docker is running
if (Test-DockerRunning) {
    Write-ColorText "Docker is running" "Green"
}
else {
    Write-ColorText "Docker is not running. Please start Docker Desktop first." "Red"
    exit 1
}

# Create Docker network if it doesn't exist
$networkExists = docker network ls | Select-String "tars-network"
if (-not $networkExists) {
    Write-ColorText "Creating Docker network: tars-network" "Yellow"
    docker network create tars-network
    Write-ColorText "Docker network created: tars-network" "Green"
}

# Create backup directory
$backupDir = "Backups\$(Get-Date -Format 'yyyyMMdd-HHmmss')"
New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
Write-ColorText "Created backup directory: $backupDir" "Green"

# Step 1: Start the swarm
Write-ColorText "Step 1: Starting TARS swarm..." "Cyan"
docker-compose -f docker-compose-swarm.yml up -d
Write-ColorText "TARS swarm started" "Green"

# Wait for the swarm to initialize
Write-ColorText "Waiting for the swarm to initialize..." "Yellow"
Start-Sleep -Seconds 10

# Step 2: Start the auto-coding process
Write-ColorText "Step 2: Starting auto-coding process..." "Cyan"
$response = Invoke-RestMethod -Uri "http://localhost:8990/api/swarm/start" -Method Post -Body (@{
    targetDirectories = $TargetDirectories
    agentCount = $AgentCount
    model = $Model
    autoApply = $false # We'll apply manually later
} | ConvertTo-Json) -ContentType "application/json"

Write-ColorText "Auto-coding process started: $($response.message)" "Green"

# Step 3: Wait for the auto-coding process to complete
Write-ColorText "Step 3: Waiting for auto-coding process to complete..." "Cyan"
$completed = $false
$timeout = 1800 # 30 minutes
$startTime = Get-Date

while (-not $completed -and ((Get-Date) - $startTime).TotalSeconds -lt $timeout) {
    try {
        $status = Invoke-RestMethod -Uri "http://localhost:8990/api/swarm/status" -Method Get
        
        Write-ColorText "Status: $($status.status)" "Yellow"
        Write-ColorText "Progress: $($status.progress)%" "Yellow"
        
        if ($status.status -eq "completed") {
            $completed = $true
        }
        else {
            Start-Sleep -Seconds 30
        }
    }
    catch {
        Write-ColorText "Error getting status: $_" "Red"
        Start-Sleep -Seconds 30
    }
}

if (-not $completed) {
    Write-ColorText "Auto-coding process timed out" "Red"
    
    # Stop the swarm
    Write-ColorText "Stopping TARS swarm..." "Yellow"
    docker-compose -f docker-compose-swarm.yml down
    Write-ColorText "TARS swarm stopped" "Green"
    
    exit 1
}

# Step 4: Run tests on the improved code
if (-not $SkipTests) {
    Write-ColorText "Step 4: Testing improved code..." "Cyan"
    & .\Scripts\Test-SwarmImprovements.ps1
    
    if ($LASTEXITCODE -ne 0) {
        Write-ColorText "Testing failed" "Red"
        
        # Stop the swarm
        Write-ColorText "Stopping TARS swarm..." "Yellow"
        docker-compose -f docker-compose-swarm.yml down
        Write-ColorText "TARS swarm stopped" "Green"
        
        exit 1
    }
    
    Write-ColorText "Testing completed successfully" "Green"
}
else {
    Write-ColorText "Step 4: Skipping tests" "Cyan"
}

# Step 5: Apply the improvements
if ($AutoApply) {
    Write-ColorText "Step 5: Applying improvements..." "Cyan"
    & .\Scripts\Apply-SwarmImprovements.ps1
    
    if ($LASTEXITCODE -ne 0) {
        Write-ColorText "Failed to apply improvements" "Red"
        
        # Stop the swarm
        Write-ColorText "Stopping TARS swarm..." "Yellow"
        docker-compose -f docker-compose-swarm.yml down
        Write-ColorText "TARS swarm stopped" "Green"
        
        exit 1
    }
    
    Write-ColorText "Improvements applied successfully" "Green"
}
else {
    Write-ColorText "Step 5: Skipping automatic application of improvements" "Cyan"
    Write-ColorText "To apply improvements, run: .\Scripts\Apply-SwarmImprovements.ps1" "Yellow"
}

# Step 6: Stop the swarm
Write-ColorText "Step 6: Stopping TARS swarm..." "Cyan"
docker-compose -f docker-compose-swarm.yml down
Write-ColorText "TARS swarm stopped" "Green"

Write-ColorText "Auto-coding process completed successfully" "Cyan"
Write-ColorText "Backup directory: $backupDir" "Green"

if (-not $AutoApply) {
    Write-ColorText "To apply improvements, run: .\Scripts\Apply-SwarmImprovements.ps1" "Yellow"
}
