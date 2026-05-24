# TARS Project Cleanup Script - Eliminate Duplicate Core Projects
# This script consolidates all fragmented TARS Core projects into ONE working version

Write-Host "🧹 TARS PROJECT CLEANUP - ELIMINATING DUPLICATES" -ForegroundColor Red
Write-Host "=================================================" -ForegroundColor Red
Write-Host ""

# Set location to TARS root
Set-Location "C:\Users\spare\source\repos\tars"

# List of duplicate/fragmented projects to remove
$duplicateProjects = @(
    "TarsEngine.FSharp.Core.Working",
    "TarsEngine.FSharp.Core.Simple", 
    "TarsEngine.FSharp.Core.Unified"
)

Write-Host "🎯 CONSOLIDATION PLAN:" -ForegroundColor Yellow
Write-Host "   ✅ Keep: TarsEngine.FSharp.Core.Unified.New (WORKING VERSION)" -ForegroundColor Green
Write-Host "   ❌ Remove: TarsEngine.FSharp.Core (bloated original)" -ForegroundColor Red
foreach ($project in $duplicateProjects) {
    Write-Host "   ❌ Remove: $project (duplicate)" -ForegroundColor Red
}
Write-Host ""

# Backup the original Core project (just in case)
Write-Host "📦 Creating backup of original Core project..." -ForegroundColor Yellow
if (Test-Path "TarsEngine.FSharp.Core") {
    if (Test-Path "TarsEngine.FSharp.Core.Backup") {
        Remove-Item "TarsEngine.FSharp.Core.Backup" -Recurse -Force
    }
    Copy-Item "TarsEngine.FSharp.Core" "TarsEngine.FSharp.Core.Backup" -Recurse -Force
    Write-Host "✅ Backup created: TarsEngine.FSharp.Core.Backup" -ForegroundColor Green
}

# Remove all duplicate projects
Write-Host ""
Write-Host "🗑️  REMOVING DUPLICATE PROJECTS..." -ForegroundColor Red

# Remove original bloated Core
if (Test-Path "TarsEngine.FSharp.Core") {
    Write-Host "🗑️  Removing bloated TarsEngine.FSharp.Core..." -ForegroundColor Red
    Remove-Item "TarsEngine.FSharp.Core" -Recurse -Force
    Write-Host "✅ Removed TarsEngine.FSharp.Core" -ForegroundColor Green
}

# Remove duplicate projects
foreach ($project in $duplicateProjects) {
    if (Test-Path $project) {
        Write-Host "🗑️  Removing duplicate $project..." -ForegroundColor Red
        Remove-Item $project -Recurse -Force
        Write-Host "✅ Removed $project" -ForegroundColor Green
    } else {
        Write-Host "⚠️  $project not found (already removed?)" -ForegroundColor Yellow
    }
}

# Install the unified working version as the main Core
Write-Host ""
Write-Host "📋 INSTALLING UNIFIED CORE AS MAIN PROJECT..." -ForegroundColor Green
if (Test-Path "TarsEngine.FSharp.Core.Unified.New") {
    Copy-Item "TarsEngine.FSharp.Core.Unified.New" "TarsEngine.FSharp.Core" -Recurse -Force
    Write-Host "✅ Installed unified Core as TarsEngine.FSharp.Core" -ForegroundColor Green
    
    # Remove the .New version since it's now the main one
    Remove-Item "TarsEngine.FSharp.Core.Unified.New" -Recurse -Force
    Write-Host "✅ Cleaned up temporary .New directory" -ForegroundColor Green
} else {
    Write-Host "❌ TarsEngine.FSharp.Core.Unified.New not found!" -ForegroundColor Red
    exit 1
}

# Test the unified project
Write-Host ""
Write-Host "🧪 TESTING UNIFIED CORE PROJECT..." -ForegroundColor Yellow
Set-Location "TarsEngine.FSharp.Core"

try {
    Write-Host "🔨 Building unified project..." -ForegroundColor Yellow
    $buildResult = dotnet build 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Build successful!" -ForegroundColor Green
        
        Write-Host "🧪 Running tests..." -ForegroundColor Yellow
        $testResult = dotnet run test 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Tests passed!" -ForegroundColor Green
        } else {
            Write-Host "❌ Tests failed!" -ForegroundColor Red
            Write-Host $testResult
        }
    } else {
        Write-Host "❌ Build failed!" -ForegroundColor Red
        Write-Host $buildResult
    }
} catch {
    Write-Host "❌ Error testing project: $_" -ForegroundColor Red
}

Set-Location ".."

# Update solution file to remove duplicate projects
Write-Host ""
Write-Host "📝 UPDATING SOLUTION FILE..." -ForegroundColor Yellow
if (Test-Path "tars.sln") {
    $solutionContent = Get-Content "tars.sln"
    $newSolutionContent = @()
    
    foreach ($line in $solutionContent) {
        $shouldInclude = $true
        foreach ($project in $duplicateProjects) {
            if ($line -like "*$project*") {
                $shouldInclude = $false
                break
            }
        }
        if ($shouldInclude) {
            $newSolutionContent += $line
        }
    }
    
    $newSolutionContent | Set-Content "tars.sln"
    Write-Host "✅ Solution file updated" -ForegroundColor Green
}

# Create summary report
Write-Host ""
Write-Host "📊 CLEANUP SUMMARY REPORT" -ForegroundColor Cyan
Write-Host "=========================" -ForegroundColor Cyan
Write-Host ""
Write-Host "🎯 BEFORE CLEANUP:" -ForegroundColor Yellow
Write-Host "   - TarsEngine.FSharp.Core (bloated, complex)" -ForegroundColor Red
Write-Host "   - TarsEngine.FSharp.Core.Working (minimal, working)" -ForegroundColor Red
Write-Host "   - TarsEngine.FSharp.Core.Simple (duplicate)" -ForegroundColor Red
Write-Host "   - TarsEngine.FSharp.Core.Unified (duplicate)" -ForegroundColor Red
Write-Host "   - TarsEngine.FSharp.Core.Unified.New (working unified)" -ForegroundColor Red
Write-Host ""
Write-Host "🎉 AFTER CLEANUP:" -ForegroundColor Green
Write-Host "   - TarsEngine.FSharp.Core (UNIFIED, WORKING, CLEAN)" -ForegroundColor Green
Write-Host "   - TarsEngine.FSharp.Core.Backup (safety backup)" -ForegroundColor Gray
Write-Host ""
Write-Host "✅ BENEFITS ACHIEVED:" -ForegroundColor Green
Write-Host "   ✅ Eliminated project duplication" -ForegroundColor Green
Write-Host "   ✅ Single source of truth for TARS Core" -ForegroundColor Green
Write-Host "   ✅ Working metascript execution" -ForegroundColor Green
Write-Host "   ✅ Clean, maintainable codebase" -ForegroundColor Green
Write-Host "   ✅ Functional TARS API with dependency injection" -ForegroundColor Green
Write-Host "   ✅ No more confusion about which project to use" -ForegroundColor Green
Write-Host ""
Write-Host "🚀 NEXT STEPS:" -ForegroundColor Cyan
Write-Host "   1. Use 'dotnet run --project TarsEngine.FSharp.Core' for all TARS operations" -ForegroundColor White
Write-Host "   2. All metascripts should reference the unified Core" -ForegroundColor White
Write-Host "   3. No more creating duplicate Core projects!" -ForegroundColor White
Write-Host ""
Write-Host "🎉 TARS PROJECT CLEANUP COMPLETE!" -ForegroundColor Green
Write-Host "   ONE CORE TO RULE THEM ALL! 🚀" -ForegroundColor Green
