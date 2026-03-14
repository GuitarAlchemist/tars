# Consolidate-Explorations.ps1
#
# This script consolidates exploration files to minimize friction with GitHub.
# It organizes explorations by topic rather than version, preserving version history.
#
# Usage: .\Consolidate-Explorations.ps1

# Configuration
$sourceDir = Join-Path $PSScriptRoot "..\docs\Explorations"
$targetDir = Join-Path $PSScriptRoot "..\docs\Explorations-Consolidated"
$metadataFile = Join-Path $targetDir "metadata.json"

# Create target directory if it doesn't exist
if (-not (Test-Path $targetDir)) {
    New-Item -ItemType Directory -Path $targetDir | Out-Null
    Write-Host "Created target directory: $targetDir"
}

# Initialize metadata
$metadata = @{}
if (Test-Path $metadataFile) {
    $metadata = Get-Content $metadataFile -Raw | ConvertFrom-Json -AsHashtable
}

# Function to extract topic from file content
function Get-TopicFromContent {
    param (
        [string]$filePath
    )
    
    $content = Get-Content $filePath -Raw
    
    # Try to extract topic from the first line or heading
    if ($content -match "^#\s+(.+)$") {
        return $matches[1]
    }
    elseif ($content -match "^(.+?)\r?\n") {
        return $matches[1]
    }
    else {
        # Use filename as fallback
        return [System.IO.Path]::GetFileNameWithoutExtension($filePath)
    }
}

# Function to sanitize topic for use as directory name
function Get-SanitizedTopicName {
    param (
        [string]$topic
    )
    
    # Remove invalid characters and trim
    $sanitized = $topic -replace '[\\/:*?"<>|]', '-'
    $sanitized = $sanitized -replace '\s+', ' '
    $sanitized = $sanitized.Trim()
    
    # Limit length
    if ($sanitized.Length -gt 50) {
        $sanitized = $sanitized.Substring(0, 50)
    }
    
    return $sanitized
}

# Process all exploration files
Write-Host "Processing exploration files..."
$explorationFiles = Get-ChildItem -Path $sourceDir -Recurse -File -Include "*.txt", "*.md"

foreach ($file in $explorationFiles) {
    # Skip metadata files
    if ($file.Name -eq "metadata.json") {
        continue
    }
    
    # Extract version from path
    $versionMatch = $file.DirectoryName -match "v(\d+)"
    $version = if ($versionMatch) { $matches[1] } else { "1" }
    
    # Extract topic from content
    $topic = Get-TopicFromContent -filePath $file.FullName
    $sanitizedTopic = Get-SanitizedTopicName -topic $topic
    
    # Create topic directory if it doesn't exist
    $topicDir = Join-Path $targetDir $sanitizedTopic
    if (-not (Test-Path $topicDir)) {
        New-Item -ItemType Directory -Path $topicDir | Out-Null
        Write-Host "Created topic directory: $topicDir"
    }
    
    # Determine target filename
    $extension = $file.Extension
    $targetFilename = "v$version$extension"
    $targetPath = Join-Path $topicDir $targetFilename
    
    # Copy file
    Copy-Item -Path $file.FullName -Destination $targetPath
    Write-Host "Copied $($file.Name) to $targetPath"
    
    # Update metadata
    if (-not $metadata.ContainsKey($sanitizedTopic)) {
        $metadata[$sanitizedTopic] = @{
            "topic" = $topic
            "versions" = @{}
            "latestVersion" = $version
            "createdDate" = (Get-Date).ToString("yyyy-MM-dd")
            "lastModifiedDate" = (Get-Date).ToString("yyyy-MM-dd")
        }
    }
    
    $metadata[$sanitizedTopic]["versions"]["v$version"] = @{
        "filePath" = $targetFilename
        "originalPath" = $file.FullName.Replace($PSScriptRoot, "").TrimStart("\")
        "createdDate" = (Get-Date).ToString("yyyy-MM-dd")
    }
    
    # Update latest version if needed
    if ([int]$version -gt [int]($metadata[$sanitizedTopic]["latestVersion"])) {
        $metadata[$sanitizedTopic]["latestVersion"] = $version
    }
    
    $metadata[$sanitizedTopic]["lastModifiedDate"] = (Get-Date).ToString("yyyy-MM-dd")
}

# Create README.md in the target directory
$readmePath = Join-Path $targetDir "README.md"
$readmeContent = @"
# TARS Explorations

This directory contains consolidated explorations organized by topic.

## Structure

Each subdirectory represents a topic and contains multiple versions of explorations on that topic.
Files are named according to their version (e.g., `v1.md`, `v2.md`).

## Topics

$(foreach ($key in $metadata.Keys) {
    "- [$($metadata[$key].topic)](./$key/v$($metadata[$key].latestVersion).md) (Latest: v$($metadata[$key].latestVersion))"
})

## Metadata

Metadata for all explorations is stored in `metadata.json`.
"@

Set-Content -Path $readmePath -Value $readmeContent
Write-Host "Created README.md in $targetDir"

# Save metadata
$metadataJson = $metadata | ConvertTo-Json -Depth 10
Set-Content -Path $metadataFile -Value $metadataJson
Write-Host "Saved metadata to $metadataFile"

Write-Host "Exploration consolidation complete!"
