# Apply-SwarmImprovements.ps1
# This script applies the improvements made by the swarm to the host

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

# Main script
Write-ColorText "TARS Swarm Improvements Application" "Cyan"
Write-ColorText "================================" "Cyan"

# Check if Docker is running
if (Test-DockerRunning) {
    Write-ColorText "Docker is running" "Green"
}
else {
    Write-ColorText "Docker is not running. Please start Docker Desktop first." "Red"
    exit 1
}

# Check if the coordinator container is running
$coordinatorRunning = docker ps | Select-String "tars-coordinator"
if (-not $coordinatorRunning) {
    Write-ColorText "Coordinator container is not running. Please start the swarm first." "Red"
    exit 1
}

# Create backup directory
$backupDir = "Backups\$(Get-Date -Format 'yyyyMMdd-HHmmss')"
New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
Write-ColorText "Created backup directory: $backupDir" "Green"

# Get the list of improved files
Write-ColorText "Getting list of improved files..." "Yellow"
$response = Invoke-RestMethod -Uri "http://localhost:8990/api/swarm/improvements" -Method Get
$improvedFiles = $response.improvedFiles

if ($improvedFiles.Count -eq 0) {
    Write-ColorText "No improved files found" "Yellow"
    exit 0
}

Write-ColorText "Found $($improvedFiles.Count) improved files" "Green"

# Apply each improved file
$appliedCount = 0
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
            
            # Copy the improved file from the Docker container to a temporary location
            $tempFile = [System.IO.Path]::GetTempFileName()
            docker cp tars-coordinator:/app/shared/improvements/$improvementPath $tempFile
            
            # Copy the temporary file to the original location
            Copy-Item -Path $tempFile -Destination $filePath -Force
            
            # Remove the temporary file
            Remove-Item -Path $tempFile -Force
            
            Write-ColorText "  Applied improvements" "Green"
            $appliedCount++
        }
        else {
            Write-ColorText "  Failed to backup file, skipping improvements" "Red"
        }
    }
    else {
        Write-ColorText "  Tests failed, skipping improvements" "Red"
    }
}

Write-ColorText "Improvements application completed" "Cyan"
Write-ColorText "Applied improvements to $appliedCount files" "Green"
Write-ColorText "Backup directory: $backupDir" "Green"

# Build and test the solution
Write-ColorText "Building and testing the solution..." "Yellow"
& .\Scripts\Build-TarsInDocker.ps1

if ($LASTEXITCODE -eq 0) {
    Write-ColorText "Build and tests succeeded" "Green"
}
else {
    Write-ColorText "Build or tests failed" "Red"
    Write-ColorText "Rolling back changes..." "Yellow"
    
    # Restore each file from backup
    foreach ($file in $improvedFiles) {
        $filePath = $file.filePath
        $backupPath = Join-Path -Path $backupDir -ChildPath "$([System.IO.Path]::GetFileName($filePath)).*"
        $backupFile = Get-ChildItem -Path $backupPath | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        
        if ($backupFile) {
            Copy-Item -Path $backupFile.FullName -Destination $filePath -Force
            Write-ColorText "  Restored $filePath from backup" "Yellow"
        }
    }
    
    Write-ColorText "Rollback completed" "Yellow"
}
