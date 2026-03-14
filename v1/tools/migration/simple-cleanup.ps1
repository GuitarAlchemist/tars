# TARS Simple Cleanup Script
# PowerShell script for immediate TARS reorganization

param(
    [switch]$DryRun = $false
)

$ErrorActionPreference = "Stop"
$startTime = Get-Date

function Write-Status {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $prefix = switch ($Level) {
        "INFO" { "[INFO]" }
        "SUCCESS" { "[SUCCESS]" }
        "WARN" { "[WARN]" }
        "ERROR" { "[ERROR]" }
        default { "[INFO]" }
    }
    Write-Host "$timestamp $prefix $Message"
}

function Create-DirectoryIfNotExists {
    param([string]$Path)
    if (!(Test-Path $Path)) {
        if ($DryRun) {
            Write-Status "DRY RUN: Would create directory: $Path" "INFO"
        } else {
            New-Item -ItemType Directory -Path $Path -Force | Out-Null
            Write-Status "Created directory: $Path" "SUCCESS"
        }
    }
}

function Move-ItemSafely {
    param([string]$Source, [string]$Destination)
    
    if (!(Test-Path $Source)) {
        Write-Status "Source not found: $Source" "WARN"
        return $false
    }
    
    if ($DryRun) {
        Write-Status "DRY RUN: Would move $Source -> $Destination" "INFO"
        return $true
    }
    
    try {
        $destDir = Split-Path $Destination -Parent
        Create-DirectoryIfNotExists $destDir
        
        if (Test-Path $Destination) {
            Write-Status "Destination exists, skipping: $Destination" "WARN"
            return $false
        }
        
        Move-Item -Path $Source -Destination $Destination -Force
        Write-Status "Moved: $Source -> $Destination" "SUCCESS"
        return $true
    } catch {
        Write-Status "Failed to move $Source -> $Destination : $($_.Exception.Message)" "ERROR"
        return $false
    }
}

function Copy-ItemSafely {
    param([string]$Source, [string]$Destination)
    
    if (!(Test-Path $Source)) {
        Write-Status "Source not found: $Source" "WARN"
        return $false
    }
    
    if ($DryRun) {
        Write-Status "DRY RUN: Would copy $Source -> $Destination" "INFO"
        return $true
    }
    
    try {
        $destDir = Split-Path $Destination -Parent
        Create-DirectoryIfNotExists $destDir
        
        Copy-Item -Path $Source -Destination $Destination -Recurse -Force
        Write-Status "Copied: $Source -> $Destination" "SUCCESS"
        return $true
    } catch {
        Write-Status "Failed to copy $Source -> $Destination : $($_.Exception.Message)" "ERROR"
        return $false
    }
}

Write-Status "Starting TARS Comprehensive Cleanup and Reorganization" "INFO"
Write-Status "Dry Run Mode: $DryRun" "INFO"

# Create backup first
$backupPath = ".tars/archive/backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
Write-Status "Creating backup at: $backupPath" "INFO"

if (!$DryRun) {
    Create-DirectoryIfNotExists $backupPath
    
    # Backup critical directories
    if (Test-Path "src") { Copy-ItemSafely "src" "$backupPath/src" }
    if (Test-Path ".tars") { Copy-ItemSafely ".tars" "$backupPath/tars_backup" }
    if (Test-Path "docs") { Copy-ItemSafely "docs" "$backupPath/docs" }
}

Write-Status "Backup completed" "SUCCESS"

# Create new directory structure
Write-Status "Creating new organizational structure..." "INFO"

$newDirectories = @(
    # Core source organization
    "src/TarsEngine.FSharp.Core",
    "src/TarsEngine.FSharp.Cli", 
    "src/TarsEngine.FSharp.Web",
    "src/TarsEngine.FSharp.Tests",
    
    # Department organization
    ".tars/departments/research/teams/university",
    ".tars/departments/research/agents",
    ".tars/departments/research/projects",
    ".tars/departments/infrastructure/teams",
    ".tars/departments/infrastructure/agents",
    ".tars/departments/qa/teams",
    ".tars/departments/qa/agents",
    ".tars/departments/qa/tests",
    ".tars/departments/ui/teams",
    ".tars/departments/ui/agents",
    ".tars/departments/operations/teams",
    ".tars/departments/operations/agents",
    
    # Evolution system
    ".tars/evolution/grammars/base",
    ".tars/evolution/grammars/evolved",
    ".tars/evolution/sessions/active",
    ".tars/evolution/sessions/completed",
    ".tars/evolution/teams",
    ".tars/evolution/results",
    
    # University system
    ".tars/university/teams/research-team",
    ".tars/university/agents/individual",
    ".tars/university/agents/specialized",
    ".tars/university/collaborations",
    ".tars/university/research",
    
    # Metascripts organization
    ".tars/metascripts/core",
    ".tars/metascripts/departments",
    ".tars/metascripts/evolution",
    ".tars/metascripts/demos",
    ".tars/metascripts/tests",
    ".tars/metascripts/templates",
    
    # System configuration
    ".tars/system/config",
    ".tars/system/logs",
    ".tars/system/monitoring",
    
    # Knowledge management
    ".tars/knowledge/base",
    ".tars/knowledge/generated",
    ".tars/knowledge/research",
    
    # Workspace
    ".tars/workspace/current",
    ".tars/workspace/experiments",
    
    # Top-level organization
    "docs/architecture",
    "docs/teams",
    "docs/agents",
    "tests/unit",
    "tests/integration",
    "demos/evolution",
    "demos/teams",
    "tools/migration",
    "tools/organization",
    "archive/legacy"
)

foreach ($dir in $newDirectories) {
    Create-DirectoryIfNotExists $dir
}

Write-Status "Created new directory structure" "SUCCESS"

# Migrate core F# projects
Write-Status "Migrating core F# projects..." "INFO"

# Move existing F# projects to src/
$coreProjects = @(
    @{ Source = "TarsEngine.FSharp.Core"; Dest = "src/TarsEngine.FSharp.Core" },
    @{ Source = "TarsEngine.FSharp.Cli"; Dest = "src/TarsEngine.FSharp.Cli" },
    @{ Source = "TarsEngine.FSharp.Web"; Dest = "src/TarsEngine.FSharp.Web" },
    @{ Source = "TarsEngine.FSharp.Metascript.Runner"; Dest = "src/TarsEngine.FSharp.Cli" }
)

foreach ($project in $coreProjects) {
    if (Test-Path $project.Source) {
        Copy-ItemSafely $project.Source $project.Dest
    }
}

# Move src/TarsEngine content to proper F# Core project
if (Test-Path "src/TarsEngine") {
    $tarsEngineFiles = Get-ChildItem "src/TarsEngine" -File
    foreach ($file in $tarsEngineFiles) {
        $destination = "src/TarsEngine.FSharp.Core/$($file.Name)"
        Copy-ItemSafely $file.FullName $destination
    }
}

Write-Status "Core projects migration completed" "SUCCESS"

# Migrate university teams and agents
Write-Status "Migrating university teams and agents..." "INFO"

if (Test-Path ".tars/university/team-config.json") {
    Move-ItemSafely ".tars/university/team-config.json" ".tars/university/teams/research-team/team-config.json"
}

if (Test-Path ".tars/agents") {
    $agentFiles = Get-ChildItem ".tars/agents" -File
    foreach ($file in $agentFiles) {
        $destination = ".tars/university/agents/individual/$($file.Name)"
        Copy-ItemSafely $file.FullName $destination
    }
}

Write-Status "University teams migration completed" "SUCCESS"

# Migrate evolution system
Write-Status "Migrating evolution system..." "INFO"

if (Test-Path ".tars/grammars") {
    $grammarFiles = Get-ChildItem ".tars/grammars" -File
    foreach ($file in $grammarFiles) {
        $destination = ".tars/evolution/grammars/base/$($file.Name)"
        Copy-ItemSafely $file.FullName $destination
    }
}

Write-Status "Evolution system migration completed" "SUCCESS"

# Organize metascripts
Write-Status "Organizing metascripts..." "INFO"

if (Test-Path ".tars/metascripts") {
    $metascriptFiles = Get-ChildItem ".tars/metascripts" -File -Filter "*.trsx"
    foreach ($file in $metascriptFiles) {
        $fileName = $file.Name.ToLower()
        $destination = ""
        
        if ($fileName.Contains("demo") -or $fileName.Contains("test")) {
            $destination = ".tars/metascripts/demos/$($file.Name)"
        } elseif ($fileName.Contains("research") -or $fileName.Contains("university")) {
            $destination = ".tars/metascripts/departments/$($file.Name)"
        } elseif ($fileName.Contains("evolution") -or $fileName.Contains("grammar")) {
            $destination = ".tars/metascripts/evolution/$($file.Name)"
        } else {
            $destination = ".tars/metascripts/core/$($file.Name)"
        }
        
        Copy-ItemSafely $file.FullName $destination
    }
}

Write-Status "Metascripts organization completed" "SUCCESS"

# Clean up obsolete files
Write-Status "Cleaning up obsolete files..." "INFO"

$obsoletePatterns = @("*.backup", "*.old", "*.tmp", "*~", "*.bak")
$cleanedCount = 0

foreach ($pattern in $obsoletePatterns) {
    $obsoleteFiles = Get-ChildItem -Path "." -File -Filter $pattern -Recurse -ErrorAction SilentlyContinue
    foreach ($file in $obsoleteFiles) {
        if ($DryRun) {
            Write-Status "DRY RUN: Would delete obsolete file: $($file.FullName)" "INFO"
            $cleanedCount++
        } else {
            try {
                Remove-Item $file.FullName -Force
                Write-Status "Deleted obsolete file: $($file.FullName)" "SUCCESS"
                $cleanedCount++
            } catch {
                Write-Status "Failed to delete $($file.FullName): $($_.Exception.Message)" "ERROR"
            }
        }
    }
}

Write-Status "Cleaned up $cleanedCount obsolete files" "SUCCESS"

# Validation
Write-Status "Validating migration results..." "INFO"

$validationChecks = @(
    @{ Path = "src"; Description = "Core source code" },
    @{ Path = ".tars/departments"; Description = "Department organization" },
    @{ Path = ".tars/evolution"; Description = "Evolution system" },
    @{ Path = ".tars/university"; Description = "University teams" },
    @{ Path = ".tars/system"; Description = "System configuration" }
)

$successCount = 0
foreach ($check in $validationChecks) {
    if (Test-Path $check.Path) {
        $fileCount = (Get-ChildItem -Path $check.Path -File -Recurse -ErrorAction SilentlyContinue | Measure-Object).Count
        Write-Status "$($check.Description): $fileCount files - OK" "SUCCESS"
        $successCount++
    } else {
        Write-Status "$($check.Description): Directory not found - ERROR" "ERROR"
    }
}

$totalChecks = $validationChecks.Count
Write-Status "Migration Validation: $successCount/$totalChecks components validated successfully" "INFO"

$endTime = Get-Date
$duration = $endTime - $startTime
Write-Status "TARS Cleanup completed in $($duration.ToString('hh\:mm\:ss'))" "SUCCESS"

if ($successCount -eq $totalChecks) {
    Write-Status "MIGRATION COMPLETED SUCCESSFULLY!" "SUCCESS"
    Write-Status "Your TARS system is now properly organized!" "SUCCESS"
} else {
    Write-Status "Migration completed with some issues - please review" "WARN"
}

Write-Status "Backup location: $backupPath" "INFO"
Write-Status "Next steps: Test core functionality and update any remaining references" "INFO"
