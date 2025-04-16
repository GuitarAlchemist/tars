# Restore-AutoCodingExample.ps1
# This script restores the original AutoCodingExample.cs file

# Function to display colored text
function Write-ColorText {
    param (
        [string]$Text,
        [string]$Color = "White"
    )
    
    Write-Host $Text -ForegroundColor $Color
}

# Main script
Write-ColorText "Restoring AutoCodingExample.cs" "Cyan"
Write-ColorText "=========================" "Cyan"

$backupFile = "TarsCli\Examples\AutoCodingExample.cs.bak"
$targetFile = "TarsCli\Examples\AutoCodingExample.cs"

if (Test-Path $backupFile) {
    Write-ColorText "Found backup file: $backupFile" "Green"
    
    # Copy the backup file to the target file
    Copy-Item -Path $backupFile -Destination $targetFile -Force
    
    Write-ColorText "Original file restored" "Green"
}
else {
    Write-ColorText "Backup file not found: $backupFile" "Red"
    exit 1
}

Write-ColorText "Restoration completed" "Cyan"
