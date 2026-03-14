# Run-SwarmAutoCode.ps1
# This script runs the swarm-based auto-coding process

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

# Function to create a backup of a file
function Backup-File {
    param (
        [string]$FilePath,
        [string]$BackupDir
    )
    
    try {
        # Create backup directory if it doesn't exist
        if (-not (Test-Path $BackupDir)) {
            New-Item -ItemType Directory -Path $BackupDir -Force | Out-Null
        }
        
        # Create backup file path
        $fileName = [System.IO.Path]::GetFileName($FilePath)
        $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
        $backupFilePath = Join-Path -Path $BackupDir -ChildPath "${fileName}.${timestamp}.bak"
        
        # Copy file to backup
        Copy-Item -Path $FilePath -Destination $backupFilePath -Force
        
        return $backupFilePath
    }
    catch {
        Write-ColorText "Error backing up file $FilePath: $_" "Red"
        return $null
    }
}

# Function to restore a file from backup
function Restore-File {
    param (
        [string]$FilePath,
        [string]$BackupFilePath
    )
    
    try {
        # Check if backup file exists
        if (-not (Test-Path $BackupFilePath)) {
            Write-ColorText "Backup file not found: $BackupFilePath" "Red"
            return $false
        }
        
        # Copy backup file to original location
        Copy-Item -Path $BackupFilePath -Destination $FilePath -Force
        
        return $true
    }
    catch {
        Write-ColorText "Error restoring file $FilePath: $_" "Red"
        return $false
    }
}

# Main script
Write-ColorText "TARS Swarm Auto-Coding" "Cyan"
Write-ColorText "====================" "Cyan"

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

# Start the swarm
Write-ColorText "Starting TARS swarm..." "Yellow"
docker-compose -f docker-compose-swarm.yml up -d
Write-ColorText "TARS swarm started" "Green"

# Wait for the swarm to initialize
Write-ColorText "Waiting for the swarm to initialize..." "Yellow"
Start-Sleep -Seconds 10

# Get target directories from command line or use defaults
$targetDirs = $args
if ($targetDirs.Count -eq 0) {
    $targetDirs = @("TarsCli", "TarsEngine")
}

Write-ColorText "Target directories: $($targetDirs -join ', ')" "Green"

# Start the auto-coding process
Write-ColorText "Starting auto-coding process..." "Yellow"
$response = Invoke-RestMethod -Uri "http://localhost:8990/api/swarm/start" -Method Post -Body (@{
    targetDirectories = $targetDirs
    agentCount = 3
    model = "llama3"
} | ConvertTo-Json) -ContentType "application/json"

Write-ColorText "Auto-coding process started: $($response.message)" "Green"

# Wait for the auto-coding process to complete
Write-ColorText "Waiting for auto-coding process to complete..." "Yellow"
$completed = $false
$timeout = 600 # 10 minutes
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
            Start-Sleep -Seconds 10
        }
    }
    catch {
        Write-ColorText "Error getting status: $_" "Red"
        Start-Sleep -Seconds 10
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

# Get the results
Write-ColorText "Auto-coding process completed" "Green"
$results = Invoke-RestMethod -Uri "http://localhost:8990/api/swarm/results" -Method Get

# Process the results
Write-ColorText "Processing results..." "Yellow"
$improvedFiles = $results.improvedFiles

foreach ($file in $improvedFiles) {
    $filePath = $file.filePath
    $improvementPath = $file.improvementPath
    $testsPassed = $file.testsPassed
    
    Write-ColorText "File: $filePath" "White"
    Write-ColorText "  Tests Passed: $testsPassed" "White"
    
    if ($testsPassed) {
        # Backup the original file
        $backupPath = Backup-File -FilePath $filePath -BackupDir $backupDir
        
        if ($backupPath) {
            Write-ColorText "  Backed up to: $backupPath" "Green"
            
            # Copy the improved file to the original location
            Copy-Item -Path $improvementPath -Destination $filePath -Force
            Write-ColorText "  Applied improvements" "Green"
        }
        else {
            Write-ColorText "  Failed to backup file, skipping improvements" "Red"
        }
    }
    else {
        Write-ColorText "  Tests failed, skipping improvements" "Red"
    }
}

# Stop the swarm
Write-ColorText "Stopping TARS swarm..." "Yellow"
docker-compose -f docker-compose-swarm.yml down
Write-ColorText "TARS swarm stopped" "Green"

Write-ColorText "Auto-coding completed" "Cyan"
Write-ColorText "Improved files: $($improvedFiles.Count)" "Green"
Write-ColorText "Backup directory: $backupDir" "Green"
