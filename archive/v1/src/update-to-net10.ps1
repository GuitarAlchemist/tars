#!/usr/bin/env pwsh

# TARS .NET 10 Migration Script
# Updates all projects to target .NET 10 for the latest features

Write-Host "🚀 TARS .NET 10 Migration Script" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""

# Function to update project file target framework
function Update-ProjectTargetFramework {
    param(
        [string]$ProjectPath,
        [string]$TargetFramework = "net10.0"
    )
    
    if (Test-Path $ProjectPath) {
        Write-Host "📝 Updating $ProjectPath..." -ForegroundColor Yellow
        
        $content = Get-Content $ProjectPath -Raw
        
        # Update various target framework patterns
        $patterns = @(
            '<TargetFramework>net9\.0</TargetFramework>',
            '<TargetFramework>net8\.0</TargetFramework>',
            '<TargetFramework>net7\.0</TargetFramework>',
            '<TargetFramework>net6\.0</TargetFramework>'
        )
        
        $updated = $false
        foreach ($pattern in $patterns) {
            if ($content -match $pattern) {
                $content = $content -replace $pattern, "<TargetFramework>$TargetFramework</TargetFramework>"
                $updated = $true
            }
        }
        
        if ($updated) {
            Set-Content -Path $ProjectPath -Value $content
            Write-Host "  ✅ Updated to $TargetFramework" -ForegroundColor Green
            return $true
        } else {
            Write-Host "  ℹ️  Already targeting $TargetFramework or no update needed" -ForegroundColor Blue
            return $false
        }
    } else {
        Write-Host "  ❌ Project file not found: $ProjectPath" -ForegroundColor Red
        return $false
    }
}

# Function to update package references for .NET 10 compatibility
function Update-PackageReferences {
    param([string]$ProjectPath)

    if (Test-Path $ProjectPath) {
        # For .NET 10 preview, we need to keep compatible package versions
        # Most packages don't have .NET 10 versions yet, so we keep .NET 9 versions
        # which are compatible with .NET 10

        # Don't update package versions for .NET 10 preview
        # The framework will handle compatibility automatically

        Write-Host "  ℹ️  Package versions kept compatible for .NET 10 preview" -ForegroundColor Blue
    }
}

# Find all project files in the repository
Write-Host "🔍 Scanning for project files..." -ForegroundColor Yellow
$projectFiles = Get-ChildItem -Path "." -Recurse -Include "*.fsproj", "*.csproj" | Where-Object { $_.FullName -notmatch "\\bin\\|\\obj\\|\\node_modules\\" }

Write-Host "Found $($projectFiles.Count) project files" -ForegroundColor Green
Write-Host ""

# Track statistics
$totalProjects = 0
$updatedProjects = 0
$skippedProjects = 0

# Update each project
foreach ($projectFile in $projectFiles) {
    $totalProjects++
    $relativePath = $projectFile.FullName.Replace((Get-Location).Path + "\", "")
    
    Write-Host "📁 Processing: $relativePath" -ForegroundColor Cyan
    
    $wasUpdated = Update-ProjectTargetFramework -ProjectPath $projectFile.FullName -TargetFramework "net10.0"
    Update-PackageReferences -ProjectPath $projectFile.FullName
    
    if ($wasUpdated) {
        $updatedProjects++
    } else {
        $skippedProjects++
    }
    
    Write-Host ""
}

# Summary
Write-Host "📊 MIGRATION SUMMARY" -ForegroundColor Cyan
Write-Host "===================" -ForegroundColor Cyan
Write-Host "Total Projects Scanned: $totalProjects" -ForegroundColor White
Write-Host "Projects Updated: $updatedProjects" -ForegroundColor Green
Write-Host "Projects Skipped (already .NET 10): $skippedProjects" -ForegroundColor Blue
Write-Host ""

Write-Host "🎯 Next Steps:" -ForegroundColor Yellow
Write-Host "1. Run 'dotnet restore' to restore packages" -ForegroundColor White
Write-Host "2. Run 'dotnet build Tars.sln -c Release' to test compilation" -ForegroundColor White
Write-Host "3. Run 'dotnet test Tars.sln -c Release' to run tests" -ForegroundColor White
Write-Host ""

Write-Host "✅ .NET 10 migration completed!" -ForegroundColor Green
