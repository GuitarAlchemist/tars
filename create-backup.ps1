# Script to create a backup of both solutions
$sourceRoot = "C:\Users\spare\source\repos\tars\Rescue\tars"
$targetRoot = "C:\Users\spare\source\repos\tars"
$backupDir = "$targetRoot\Backups\Migration-$(Get-Date -Format 'yyyy-MM-dd_HH-mm-ss')"

# Create backup directory
Write-Host "Creating backup directory: $backupDir"
New-Item -ItemType Directory -Path $backupDir -Force | Out-Null

# Backup the target solution file
Write-Host "Backing up target solution file..."
Copy-Item -Path "$targetRoot\tars.sln" -Destination "$backupDir\tars.sln" -Force

# Create source and target backup directories
New-Item -ItemType Directory -Path "$backupDir\source" -Force | Out-Null
New-Item -ItemType Directory -Path "$backupDir\target" -Force | Out-Null

# Backup key source directories
$sourceDirectories = @(
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
    "TarsApp"
)

foreach ($dir in $sourceDirectories) {
    $sourcePath = "$sourceRoot\$dir"
    if (Test-Path $sourcePath) {
        Write-Host "Backing up source directory: $dir"
        Copy-Item -Path $sourcePath -Destination "$backupDir\source\$dir" -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# Backup key target directories
$targetDirectories = @(
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
    "TarsCli.Core",
    "TarsCli.Commands",
    "TarsCli.Models",
    "TarsCli.Services",
    "TarsCli.Mcp",
    "TarsCli.Docker",
    "TarsCli.DSL",
    "TarsCli.CodeAnalysis",
    "TarsCli.Testing",
    "TarsCli.WebUI",
    "TarsCli.App",
    "TarsEngine.Intelligence",
    "TarsEngine.ML",
    "TarsEngine.Consciousness",
    "TarsEngine.A2A",
    "TarsEngine.Metrics",
    "TarsEngine.Monads",
    "TarsEngine.Services",
    "TarsApp"
)

foreach ($dir in $targetDirectories) {
    $targetPath = "$targetRoot\$dir"
    if (Test-Path $targetPath) {
        Write-Host "Backing up target directory: $dir"
        Copy-Item -Path $targetPath -Destination "$backupDir\target\$dir" -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "Backup completed successfully to: $backupDir"
