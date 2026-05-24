# Script to merge projects from Rescue solution to target solution
$sourceRoot = "C:\Users\spare\source\repos\tars\Rescue\tars"
$targetRoot = "C:\Users\spare\source\repos\tars"

# Common projects to copy directly
$commonProjects = @(
    "TarsEngine",
    "TarsEngineFSharp",
    "TarsEngine.Interfaces",
    "TarsEngine.SelfImprovement",
    "TarsEngine.DSL",
    "TarsEngine.DSL.Tests",
    "TarsEngine.Tests",
    "TarsEngineFSharp.Core",
    "TarsEngine.Unified",
    "TarsCli",
    "TarsApp",
    "Experiments\ChatbotExample1"
)

# Create backup of target solution
$backupDir = "$targetRoot\Backups\$(Get-Date -Format 'yyyy-MM-dd_HH-mm-ss')"
New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
Copy-Item -Path "$targetRoot\tars.sln" -Destination "$backupDir\tars.sln" -Force

# Copy common projects
foreach ($project in $commonProjects) {
    $sourcePath = "$sourceRoot\$project"
    $targetPath = "$targetRoot\$project"
    
    if (Test-Path $sourcePath) {
        Write-Host "Copying $project..."
        
        # Create target directory if it doesn't exist
        if (-not (Test-Path $targetPath)) {
            New-Item -ItemType Directory -Path $targetPath -Force | Out-Null
        }
        
        # Copy all files except bin and obj directories
        Get-ChildItem -Path $sourcePath -Recurse -Exclude "bin", "obj" | 
        ForEach-Object {
            $relativePath = $_.FullName.Substring($sourcePath.Length)
            $destination = Join-Path $targetPath $relativePath
            
            if ($_.PSIsContainer) {
                if (-not (Test-Path $destination)) {
                    New-Item -ItemType Directory -Path $destination -Force | Out-Null
                }
            } else {
                $destinationFolder = Split-Path -Path $destination -Parent
                if (-not (Test-Path $destinationFolder)) {
                    New-Item -ItemType Directory -Path $destinationFolder -Force | Out-Null
                }
                Copy-Item -Path $_.FullName -Destination $destination -Force
            }
        }
    } else {
        Write-Host "Warning: Source project $project not found at $sourcePath"
    }
}

# Copy additional directories that might be needed
$additionalDirs = @(
    "docs",
    "shared",
    "config",
    "data",
    "templates"
)

foreach ($dir in $additionalDirs) {
    $sourcePath = "$sourceRoot\$dir"
    $targetPath = "$targetRoot\$dir"
    
    if (Test-Path $sourcePath) {
        Write-Host "Copying $dir..."
        
        # Create target directory if it doesn't exist
        if (-not (Test-Path $targetPath)) {
            New-Item -ItemType Directory -Path $targetPath -Force | Out-Null
        }
        
        # Copy all files
        Get-ChildItem -Path $sourcePath -Recurse | 
        ForEach-Object {
            $relativePath = $_.FullName.Substring($sourcePath.Length)
            $destination = Join-Path $targetPath $relativePath
            
            if ($_.PSIsContainer) {
                if (-not (Test-Path $destination)) {
                    New-Item -ItemType Directory -Path $destination -Force | Out-Null
                }
            } else {
                $destinationFolder = Split-Path -Path $destination -Parent
                if (-not (Test-Path $destinationFolder)) {
                    New-Item -ItemType Directory -Path $destinationFolder -Force | Out-Null
                }
                Copy-Item -Path $_.FullName -Destination $destination -Force
            }
        }
    }
}

Write-Host "Project merge completed. Please check the solution in Visual Studio to ensure all references are correct."
