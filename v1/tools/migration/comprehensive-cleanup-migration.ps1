# TARS Comprehensive Cleanup & Migration Script
# PowerShell script to support the comprehensive reorganization

param(
    [string]$Phase = "all",
    [switch]$DryRun = $false,
    [switch]$Verbose = $false
)

$ErrorActionPreference = "Stop"
$startTime = Get-Date
$logPath = ".tars/system/logs/migration_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

# Ensure log directory exists
$logDir = Split-Path $logPath -Parent
if (!(Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logEntry = "[$timestamp] [$Level] $Message"
    Write-Host $logEntry
    Add-Content -Path $logPath -Value $logEntry
}

function Test-PathSafely {
    param([string]$Path)
    try {
        return Test-Path $Path
    } catch {
        return $false
    }
}

function Move-FileSafely {
    param(
        [string]$Source,
        [string]$Destination,
        [switch]$Force = $false
    )
    
    if ($DryRun) {
        Write-Log "DRY RUN: Would move $Source -> $Destination" "DRYRUN"
        return $true
    }
    
    try {
        $destDir = Split-Path $Destination -Parent
        if (!(Test-Path $destDir)) {
            New-Item -ItemType Directory -Path $destDir -Force | Out-Null
        }
        
        if (Test-Path $Destination) {
            if ($Force) {
                Remove-Item $Destination -Force
            } else {
                Write-Log "Destination exists, skipping: $Destination" "WARN"
                return $false
            }
        }
        
        Move-Item -Path $Source -Destination $Destination -Force
        Write-Log "Moved: $Source -> $Destination" "SUCCESS"
        return $true
    } catch {
        Write-Log "Failed to move $Source -> $Destination : $($_.Exception.Message)" "ERROR"
        return $false
    }
}

function Copy-DirectorySafely {
    param(
        [string]$Source,
        [string]$Destination
    )
    
    if (!(Test-PathSafely $Source)) {
        Write-Log "Source directory not found: $Source" "WARN"
        return $false
    }
    
    if ($DryRun) {
        Write-Log "DRY RUN: Would copy directory $Source -> $Destination" "DRYRUN"
        return $true
    }
    
    try {
        if (!(Test-Path $Destination)) {
            New-Item -ItemType Directory -Path $Destination -Force | Out-Null
        }
        
        Copy-Item -Path "$Source\*" -Destination $Destination -Recurse -Force
        Write-Log "Copied directory: $Source -> $Destination" "SUCCESS"
        return $true
    } catch {
        Write-Log "Failed to copy directory $Source -> $Destination : $($_.Exception.Message)" "ERROR"
        return $false
    }
}

function Remove-EmptyDirectories {
    param([string]$Path)
    
    if (!(Test-PathSafely $Path)) {
        return
    }
    
    Get-ChildItem -Path $Path -Directory -Recurse | 
    Where-Object { (Get-ChildItem $_.FullName -Force | Measure-Object).Count -eq 0 } |
    ForEach-Object {
        if ($DryRun) {
            Write-Log "DRY RUN: Would remove empty directory: $($_.FullName)" "DRYRUN"
        } else {
            try {
                Remove-Item $_.FullName -Force
                Write-Log "Removed empty directory: $($_.FullName)" "SUCCESS"
            } catch {
                Write-Log "Failed to remove empty directory $($_.FullName): $($_.Exception.Message)" "ERROR"
            }
        }
    }
}

function Analyze-CurrentStructure {
    Write-Log "🔍 Analyzing current directory structure..." "INFO"
    
    # Analyze root directory
    $rootFiles = Get-ChildItem -Path "." -File | Measure-Object
    $rootDirs = Get-ChildItem -Path "." -Directory | Measure-Object
    
    Write-Log "📊 Root Directory Analysis:" "INFO"
    Write-Log "   Files: $($rootFiles.Count)" "INFO"
    Write-Log "   Directories: $($rootDirs.Count)" "INFO"
    
    # Analyze .tars directory
    if (Test-PathSafely ".tars") {
        $tarsFiles = Get-ChildItem -Path ".tars" -File -Recurse | Measure-Object
        $tarsDirs = Get-ChildItem -Path ".tars" -Directory -Recurse | Measure-Object
        
        Write-Log "📊 .tars Directory Analysis:" "INFO"
        Write-Log "   Files: $($tarsFiles.Count)" "INFO"
        Write-Log "   Directories: $($tarsDirs.Count)" "INFO"
    }
    
    # Find problematic files
    $demoFiles = Get-ChildItem -Path "." -File | Where-Object { 
        $_.Name -match "(demo|test|example)" 
    }
    
    if ($demoFiles.Count -gt 0) {
        Write-Log "⚠️  Found $($demoFiles.Count) scattered demo/test files in root" "WARN"
    }
    
    # Find duplicate files
    $allFiles = Get-ChildItem -Path "." -File -Recurse
    $duplicateGroups = $allFiles | Group-Object { [System.IO.Path]::GetFileNameWithoutExtension($_.Name) } | 
                      Where-Object { $_.Count -gt 1 }
    
    if ($duplicateGroups.Count -gt 0) {
        Write-Log "⚠️  Found $($duplicateGroups.Count) groups of potential duplicate files" "WARN"
    }
    
    Write-Log "✅ Structure analysis complete" "SUCCESS"
}

function Create-NewStructure {
    Write-Log "🏗️  Creating new organizational structure..." "INFO"
    
    $newDirectories = @(
        # Core source organization
        "src/TarsEngine.FSharp.Core",
        "src/TarsEngine.FSharp.Cli",
        "src/TarsEngine.FSharp.Web",
        "src/TarsEngine.FSharp.Tests",
        
        # Department organization
        ".tars/departments/research/teams/university",
        ".tars/departments/research/teams/innovation",
        ".tars/departments/research/teams/analysis",
        ".tars/departments/research/agents",
        ".tars/departments/research/projects",
        ".tars/departments/research/reports",
        
        ".tars/departments/infrastructure/teams",
        ".tars/departments/infrastructure/agents",
        ".tars/departments/infrastructure/deployment",
        ".tars/departments/infrastructure/monitoring",
        
        ".tars/departments/qa/teams",
        ".tars/departments/qa/agents",
        ".tars/departments/qa/tests",
        ".tars/departments/qa/reports",
        
        ".tars/departments/ui/teams",
        ".tars/departments/ui/agents",
        ".tars/departments/ui/interfaces",
        ".tars/departments/ui/demos",
        
        ".tars/departments/operations/teams",
        ".tars/departments/operations/agents",
        ".tars/departments/operations/workflows",
        ".tars/departments/operations/automation",
        
        # Evolution system organization
        ".tars/evolution/grammars/base",
        ".tars/evolution/grammars/evolved",
        ".tars/evolution/grammars/templates",
        ".tars/evolution/sessions/active",
        ".tars/evolution/sessions/completed",
        ".tars/evolution/sessions/archived",
        ".tars/evolution/teams",
        ".tars/evolution/results",
        ".tars/evolution/monitoring",
        
        # University system organization
        ".tars/university/teams/research-team",
        ".tars/university/teams/cs-researchers",
        ".tars/university/teams/data-scientists",
        ".tars/university/teams/academic-writers",
        ".tars/university/agents/individual",
        ".tars/university/agents/specialized",
        ".tars/university/agents/collaborative",
        ".tars/university/collaborations",
        ".tars/university/research",
        ".tars/university/publications",
        
        # System configuration
        ".tars/system/config/departments",
        ".tars/system/config/teams",
        ".tars/system/config/agents",
        ".tars/system/config/evolution",
        ".tars/system/logs/departments",
        ".tars/system/logs/evolution",
        ".tars/system/logs/system",
        ".tars/system/monitoring/performance",
        ".tars/system/monitoring/health",
        ".tars/system/monitoring/evolution",
        ".tars/system/security",
        
        # Organized content
        ".tars/metascripts/core",
        ".tars/metascripts/departments",
        ".tars/metascripts/evolution",
        ".tars/metascripts/demos",
        ".tars/metascripts/tests",
        ".tars/metascripts/templates",
        
        ".tars/closures/evolutionary",
        ".tars/closures/traditional",
        ".tars/closures/templates",
        ".tars/closures/registry",
        
        ".tars/knowledge/base",
        ".tars/knowledge/generated",
        ".tars/knowledge/research",
        ".tars/knowledge/documentation",
        
        ".tars/workspace/current",
        ".tars/workspace/experiments",
        ".tars/workspace/collaborations",
        ".tars/workspace/staging",
        
        # Top-level organization
        "docs/architecture",
        "docs/departments",
        "docs/teams",
        "docs/agents",
        "docs/evolution",
        "docs/api",
        "docs/tutorials",
        
        "tests/unit",
        "tests/integration",
        "tests/evolution",
        "tests/departments",
        "tests/performance",
        
        "demos/evolution",
        "demos/departments",
        "demos/teams",
        "demos/agents",
        "demos/comprehensive",
        
        "tools/migration",
        "tools/organization",
        "tools/monitoring",
        "tools/maintenance",
        
        "archive/legacy-csharp",
        "archive/old-demos",
        "archive/obsolete-tests",
        "archive/backup-configs",
        "archive/historical"
    )
    
    $createdCount = 0
    foreach ($dir in $newDirectories) {
        if ($DryRun) {
            Write-Log "DRY RUN: Would create directory: $dir" "DRYRUN"
            $createdCount++
        } else {
            try {
                if (!(Test-Path $dir)) {
                    New-Item -ItemType Directory -Path $dir -Force | Out-Null
                    $createdCount++
                }
            } catch {
                Write-Log "Failed to create directory $dir : $($_.Exception.Message)" "ERROR"
            }
        }
    }
    
    Write-Log "✅ Created $createdCount new directories" "SUCCESS"
}

function Migrate-CoreSystem {
    Write-Log "🔄 Migrating core system components..." "INFO"
    
    # Migrate F# projects
    $coreProjects = @(
        @{ Source = "TarsEngine.FSharp.Core"; Dest = "src/TarsEngine.FSharp.Core" },
        @{ Source = "TarsEngine.FSharp.Cli"; Dest = "src/TarsEngine.FSharp.Cli" },
        @{ Source = "TarsEngine.FSharp.Web"; Dest = "src/TarsEngine.FSharp.Web" },
        @{ Source = "TarsEngine.FSharp.Metascript.Runner"; Dest = "src/TarsEngine.FSharp.Cli" }
    )
    
    foreach ($project in $coreProjects) {
        if (Test-PathSafely $project.Source) {
            Copy-DirectorySafely $project.Source $project.Dest
        }
    }
    
    Write-Log "✅ Core system migration complete" "SUCCESS"
}

function Migrate-UniversityTeams {
    Write-Log "🎓 Migrating university teams and agents..." "INFO"
    
    # Migrate university team configuration
    if (Test-PathSafely ".tars/university/team-config.json") {
        Move-FileSafely ".tars/university/team-config.json" ".tars/university/teams/research-team/team-config.json"
    }
    
    # Migrate agent configurations
    if (Test-PathSafely ".tars/agents") {
        Copy-DirectorySafely ".tars/agents" ".tars/university/agents/individual"
    }
    
    Write-Log "✅ University teams migration complete" "SUCCESS"
}

function Migrate-EvolutionSystem {
    Write-Log "🧬 Migrating evolution system..." "INFO"
    
    # Migrate grammars
    if (Test-PathSafely ".tars/grammars") {
        $grammarFiles = Get-ChildItem -Path ".tars/grammars" -File
        foreach ($file in $grammarFiles) {
            $destination = ".tars/evolution/grammars/base/$($file.Name)"
            Move-FileSafely $file.FullName $destination
        }
    }
    
    # Migrate evolution sessions
    if (Test-PathSafely ".tars/evolution") {
        $sessionFiles = Get-ChildItem -Path ".tars/evolution" -File -Filter "*.json"
        foreach ($file in $sessionFiles) {
            $destination = ".tars/evolution/sessions/active/$($file.Name)"
            Move-FileSafely $file.FullName $destination
        }
    }
    
    Write-Log "✅ Evolution system migration complete" "SUCCESS"
}

function Cleanup-ObsoleteFiles {
    Write-Log "🧹 Cleaning up obsolete files..." "INFO"
    
    $obsoletePatterns = @("*.backup", "*.old", "*.tmp", "*~", "*.bak", "Thumbs.db", ".DS_Store")
    $cleanedCount = 0
    
    foreach ($pattern in $obsoletePatterns) {
        $obsoleteFiles = Get-ChildItem -Path "." -File -Filter $pattern -Recurse -ErrorAction SilentlyContinue
        foreach ($file in $obsoleteFiles) {
            if ($DryRun) {
                Write-Log "DRY RUN: Would delete obsolete file: $($file.FullName)" "DRYRUN"
                $cleanedCount++
            } else {
                try {
                    Remove-Item $file.FullName -Force
                    Write-Log "Deleted obsolete file: $($file.FullName)" "SUCCESS"
                    $cleanedCount++
                } catch {
                    Write-Log "Failed to delete $($file.FullName): $($_.Exception.Message)" "ERROR"
                }
            }
        }
    }
    
    Write-Log "✅ Cleaned up $cleanedCount obsolete files" "SUCCESS"
}

function Validate-Migration {
    Write-Log "✅ Validating migration results..." "INFO"
    
    $validationChecks = @(
        @{ Path = "src"; MinFiles = 10; Description = "Core source code" },
        @{ Path = ".tars/departments"; MinFiles = 5; Description = "Department organization" },
        @{ Path = ".tars/evolution"; MinFiles = 3; Description = "Evolution system" },
        @{ Path = ".tars/university"; MinFiles = 3; Description = "University teams" },
        @{ Path = ".tars/system"; MinFiles = 3; Description = "System configuration" }
    )
    
    $successCount = 0
    foreach ($check in $validationChecks) {
        if (Test-PathSafely $check.Path) {
            $fileCount = (Get-ChildItem -Path $check.Path -File -Recurse | Measure-Object).Count
            $isValid = $fileCount -ge $check.MinFiles
            $status = if ($isValid) { "✅" } else { "❌" }
            Write-Log "   $($check.Description): $fileCount files (expected >= $($check.MinFiles)) - $status" "INFO"
            if ($isValid) { $successCount++ }
        } else {
            Write-Log "   $($check.Description): Directory not found - ❌" "ERROR"
        }
    }
    
    $totalChecks = $validationChecks.Count
    Write-Log "📊 Migration Validation: $successCount/$totalChecks components validated successfully" "INFO"
    
    if ($successCount -eq $totalChecks) {
        Write-Log "🎉 MIGRATION COMPLETED SUCCESSFULLY!" "SUCCESS"
    } else {
        Write-Log "⚠️  Migration completed with some issues - review logs" "WARN"
    }
}

# Main execution
Write-Log "🚀 Starting TARS Comprehensive Cleanup & Migration" "INFO"
Write-Log "📋 Phase: $Phase" "INFO"
Write-Log "🔍 Dry Run: $DryRun" "INFO"

try {
    switch ($Phase.ToLower()) {
        "analyze" { Analyze-CurrentStructure }
        "structure" { Create-NewStructure }
        "core" { Migrate-CoreSystem }
        "teams" { Migrate-UniversityTeams }
        "evolution" { Migrate-EvolutionSystem }
        "cleanup" { Cleanup-ObsoleteFiles }
        "validate" { Validate-Migration }
        "all" {
            Analyze-CurrentStructure
            Create-NewStructure
            Migrate-CoreSystem
            Migrate-UniversityTeams
            Migrate-EvolutionSystem
            Cleanup-ObsoleteFiles
            Remove-EmptyDirectories "."
            Validate-Migration
        }
        default {
            Write-Log "Unknown phase: $Phase. Valid phases: analyze, structure, core, teams, evolution, cleanup, validate, all" "ERROR"
            exit 1
        }
    }
    
    $endTime = Get-Date
    $duration = $endTime - $startTime
    Write-Log "🎉 Migration phase '$Phase' completed in $($duration.ToString('hh\:mm\:ss'))" "SUCCESS"
    Write-Log "📝 Full log available at: $logPath" "INFO"
    
} catch {
    Write-Log "💥 Migration failed: $($_.Exception.Message)" "ERROR"
    Write-Log "📝 Check log for details: $logPath" "ERROR"
    exit 1
}
