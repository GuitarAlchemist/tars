# TARS Cleanup Utility
# Cleans up build artifacts, temporary files, and logs

param(
    [switch]$BuildArtifacts = $false,
    [switch]$TempFiles = $false,
    [switch]$Logs = $false,
    [switch]$All = $false,
    [switch]$DryRun = $false,
    [switch]$Verbose = $false
)

Write-Host "🧹 TARS Cleanup Utility" -ForegroundColor Cyan
Write-Host "=======================" -ForegroundColor Cyan
Write-Host ""

# Set verbose preference
if ($Verbose) {
    $VerbosePreference = "Continue"
}

$CleanupStats = @{
    FilesDeleted = 0
    DirectoriesDeleted = 0
    SpaceFreed = 0
    Errors = 0
}

function Get-FolderSize {
    param([string]$Path)
    
    if (Test-Path $Path) {
        return (Get-ChildItem -Path $Path -Recurse -File | Measure-Object -Property Length -Sum).Sum
    }
    return 0
}

function Remove-ItemSafely {
    param(
        [string]$Path,
        [string]$Description
    )
    
    if (Test-Path $Path) {
        try {
            $sizeBefore = if (Test-Path $Path -PathType Container) { Get-FolderSize $Path } else { (Get-Item $Path).Length }
            
            if ($DryRun) {
                Write-Host "   [DRY RUN] Would delete: $Path" -ForegroundColor Yellow
                return
            }
            
            if (Test-Path $Path -PathType Container) {
                Remove-Item -Path $Path -Recurse -Force
                $CleanupStats.DirectoriesDeleted++
                Write-Host "   ✅ Deleted directory: $Description" -ForegroundColor Green
            } else {
                Remove-Item -Path $Path -Force
                $CleanupStats.FilesDeleted++
                Write-Host "   ✅ Deleted file: $Description" -ForegroundColor Green
            }
            
            $CleanupStats.SpaceFreed += $sizeBefore
        } catch {
            Write-Host "   ❌ Failed to delete $Description`: $($_.Exception.Message)" -ForegroundColor Red
            $CleanupStats.Errors++
        }
    } else {
        Write-Host "   ℹ️ Not found: $Description" -ForegroundColor Gray
    }
}

function Cleanup-BuildArtifacts {
    Write-Host "🔨 Cleaning Build Artifacts..." -ForegroundColor Yellow
    
    $buildPaths = @(
        @{ Path = "TarsEngine.FSharp.Cli\bin"; Description = "CLI bin directory" },
        @{ Path = "TarsEngine.FSharp.Cli\obj"; Description = "CLI obj directory" },
        @{ Path = "TarsEngine.FSharp.Metascripts\bin"; Description = "Metascripts bin directory" },
        @{ Path = "TarsEngine.FSharp.Metascripts\obj"; Description = "Metascripts obj directory" },
        @{ Path = "build"; Description = "Build output directory" },
        @{ Path = "packages"; Description = "Packages directory" }
    )
    
    foreach ($item in $buildPaths) {
        Remove-ItemSafely -Path $item.Path -Description $item.Description
    }
}

function Cleanup-TempFiles {
    Write-Host "🗂️ Cleaning Temporary Files..." -ForegroundColor Yellow
    
    $tempPaths = @(
        @{ Path = "temp"; Description = "Temp directory" },
        @{ Path = "tmp"; Description = "Tmp directory" },
        @{ Path = "*.tmp"; Description = "Temporary files" },
        @{ Path = "*.temp"; Description = "Temp files" },
        @{ Path = ".vs"; Description = "Visual Studio cache" },
        @{ Path = "*.user"; Description = "User settings files" },
        @{ Path = "*.suo"; Description = "Solution user options" },
        @{ Path = "Thumbs.db"; Description = "Windows thumbnail cache" },
        @{ Path = ".DS_Store"; Description = "macOS metadata files" }
    )
    
    foreach ($item in $tempPaths) {
        if ($item.Path -like "*.*") {
            # Handle wildcard patterns
            Get-ChildItem -Path . -Filter $item.Path -Recurse -Force -ErrorAction SilentlyContinue | ForEach-Object {
                Remove-ItemSafely -Path $_.FullName -Description "$($item.Description) ($($_.Name))"
            }
        } else {
            Remove-ItemSafely -Path $item.Path -Description $item.Description
        }
    }
}

function Cleanup-Logs {
    Write-Host "📝 Cleaning Log Files..." -ForegroundColor Yellow
    
    $logPaths = @(
        @{ Path = "logs"; Description = "Logs directory" },
        @{ Path = "*.log"; Description = "Log files" },
        @{ Path = "*.log.*"; Description = "Rotated log files" },
        @{ Path = "msbuild.log"; Description = "MSBuild log" },
        @{ Path = "build.log"; Description = "Build log" }
    )
    
    foreach ($item in $logPaths) {
        if ($item.Path -like "*.*") {
            # Handle wildcard patterns
            Get-ChildItem -Path . -Filter $item.Path -Recurse -Force -ErrorAction SilentlyContinue | ForEach-Object {
                Remove-ItemSafely -Path $_.FullName -Description "$($item.Description) ($($_.Name))"
            }
        } else {
            Remove-ItemSafely -Path $item.Path -Description $item.Description
        }
    }
}

function Cleanup-NuGetCache {
    Write-Host "📦 Cleaning NuGet Cache..." -ForegroundColor Yellow
    
    try {
        if ($DryRun) {
            Write-Host "   [DRY RUN] Would clear NuGet cache" -ForegroundColor Yellow
        } else {
            dotnet nuget locals all --clear
            Write-Host "   ✅ NuGet cache cleared" -ForegroundColor Green
        }
    } catch {
        Write-Host "   ❌ Failed to clear NuGet cache: $($_.Exception.Message)" -ForegroundColor Red
        $CleanupStats.Errors++
    }
}

function Show-CleanupSummary {
    Write-Host ""
    Write-Host "📊 Cleanup Summary" -ForegroundColor Cyan
    Write-Host "==================" -ForegroundColor Cyan
    Write-Host ""
    
    $spaceFreedMB = [math]::Round($CleanupStats.SpaceFreed / 1MB, 2)
    
    Write-Host "Files Deleted: $($CleanupStats.FilesDeleted)" -ForegroundColor White
    Write-Host "Directories Deleted: $($CleanupStats.DirectoriesDeleted)" -ForegroundColor White
    Write-Host "Space Freed: $spaceFreedMB MB" -ForegroundColor White
    Write-Host "Errors: $($CleanupStats.Errors)" -ForegroundColor $(if ($CleanupStats.Errors -eq 0) { "Green" } else { "Red" })
    
    if ($DryRun) {
        Write-Host ""
        Write-Host "ℹ️ This was a dry run. No files were actually deleted." -ForegroundColor Yellow
        Write-Host "Run without -DryRun to perform actual cleanup." -ForegroundColor Yellow
    } else {
        Write-Host ""
        if ($CleanupStats.Errors -eq 0) {
            Write-Host "✅ Cleanup completed successfully!" -ForegroundColor Green
        } else {
            Write-Host "⚠️ Cleanup completed with $($CleanupStats.Errors) errors." -ForegroundColor Yellow
        }
    }
}

# Main cleanup logic
Write-Host "🎯 Starting TARS Cleanup" -ForegroundColor Green

if ($DryRun) {
    Write-Host "🔍 DRY RUN MODE - No files will be deleted" -ForegroundColor Yellow
}

Write-Host ""

# Determine what to clean
$cleanBuild = $BuildArtifacts -or $All
$cleanTemp = $TempFiles -or $All
$cleanLogs = $Logs -or $All

# If no specific options, show help
if (-not ($cleanBuild -or $cleanTemp -or $cleanLogs)) {
    Write-Host "❓ No cleanup options specified. Available options:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "   -BuildArtifacts  Clean build artifacts (bin, obj, build directories)"
    Write-Host "   -TempFiles       Clean temporary files and caches"
    Write-Host "   -Logs            Clean log files"
    Write-Host "   -All             Clean everything"
    Write-Host "   -DryRun          Show what would be deleted without deleting"
    Write-Host "   -Verbose         Show detailed output"
    Write-Host ""
    Write-Host "Example: .\cleanup.ps1 -All -DryRun"
    exit 0
}

# Perform cleanup operations
if ($cleanBuild) {
    Cleanup-BuildArtifacts
    Cleanup-NuGetCache
}

if ($cleanTemp) {
    Cleanup-TempFiles
}

if ($cleanLogs) {
    Cleanup-Logs
}

Show-CleanupSummary
