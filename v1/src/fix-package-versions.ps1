#!/usr/bin/env pwsh

# Fix Package Versions for .NET 10 Compatibility
# Reverts package versions to compatible versions for .NET 10 preview

Write-Host "🔧 Fixing Package Versions for .NET 10 Compatibility" -ForegroundColor Cyan
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host ""

# Function to fix package versions in project files
function Fix-PackageVersions {
    param([string]$ProjectPath)
    
    if (Test-Path $ProjectPath) {
        Write-Host "📝 Fixing packages in $ProjectPath..." -ForegroundColor Yellow
        
        $content = Get-Content $ProjectPath -Raw
        $updated = $false
        
        # Revert problematic package versions to compatible ones
        $packageFixes = @{
            'Microsoft\.Extensions\.Logging' = '9.0.0'
            'Microsoft\.Extensions\.DependencyInjection' = '9.0.0'
            'Microsoft\.Extensions\.Http' = '9.0.0'
            'Microsoft\.Extensions\.Hosting' = '9.0.0'
            'Microsoft\.Extensions\.Logging\.Console' = '9.0.0'
            'System\.Text\.Json' = '9.0.0'
            'System\.Threading\.Channels' = '9.0.0'
            'System\.ServiceProcess\.ServiceController' = '9.0.0'
            'System\.Management' = '9.0.0'
            'FSharp\.Core' = '8.0.400'
        }
        
        foreach ($package in $packageFixes.Keys) {
            $correctVersion = $packageFixes[$package]
            
            # Fix version 10.0.0 back to compatible version
            if ($content -match "<PackageReference Include=`"$package`" Version=`"10\.0\.0`"") {
                $content = $content -replace "<PackageReference Include=`"$package`" Version=`"10\.0\.0`"", "<PackageReference Include=`"$package`" Version=`"$correctVersion`""
                $updated = $true
                Write-Host "  ✅ Fixed $package to version $correctVersion" -ForegroundColor Green
            }
        }
        
        if ($updated) {
            Set-Content -Path $ProjectPath -Value $content
            Write-Host "  📦 Package versions fixed" -ForegroundColor Green
        } else {
            Write-Host "  ℹ️  No package fixes needed" -ForegroundColor Blue
        }
        
        Write-Host ""
    }
}

# Find all project files in the repository
Write-Host "🔍 Scanning for project files..." -ForegroundColor Yellow
$projectFiles = Get-ChildItem -Path "." -Recurse -Include "*.fsproj", "*.csproj" | Where-Object { $_.FullName -notmatch "\\bin\\|\\obj\\|\\node_modules\\" }

Write-Host "Found $($projectFiles.Count) project files" -ForegroundColor Green
Write-Host ""

# Track statistics
$totalProjects = 0
$fixedProjects = 0

# Fix each project
foreach ($projectFile in $projectFiles) {
    $totalProjects++
    $relativePath = $projectFile.FullName.Replace((Get-Location).Path + "\", "")
    
    Write-Host "📁 Processing: $relativePath" -ForegroundColor Cyan
    Fix-PackageVersions -ProjectPath $projectFile.FullName
    $fixedProjects++
}

# Summary
Write-Host "📊 PACKAGE FIX SUMMARY" -ForegroundColor Cyan
Write-Host "======================" -ForegroundColor Cyan
Write-Host "Total Projects Processed: $totalProjects" -ForegroundColor White
Write-Host "Projects Fixed: $fixedProjects" -ForegroundColor Green
Write-Host ""

Write-Host "🎯 Next Steps:" -ForegroundColor Yellow
Write-Host "1. Run 'dotnet restore Tars.sln' to restore packages" -ForegroundColor White
Write-Host "2. Run 'dotnet build Tars.sln -c Release' to test compilation" -ForegroundColor White
Write-Host ""

Write-Host "✅ Package version fixes completed!" -ForegroundColor Green
