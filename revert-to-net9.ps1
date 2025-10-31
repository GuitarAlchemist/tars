#!/usr/bin/env pwsh

# Revert TARS projects back to .NET 9
# Since .NET 10 preview doesn't have all packages available yet

Write-Host "🔄 Reverting TARS projects back to .NET 9" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Function to revert project file target framework
function Revert-ProjectTargetFramework {
    param(
        [string]$ProjectPath,
        [string]$TargetFramework = "net9.0"
    )
    
    if (Test-Path $ProjectPath) {
        Write-Host "📝 Reverting $ProjectPath..." -ForegroundColor Yellow
        
        $content = Get-Content $ProjectPath -Raw
        
        # Revert net10.0 back to net9.0
        if ($content -match '<TargetFramework>net10\.0</TargetFramework>') {
            $content = $content -replace '<TargetFramework>net10\.0</TargetFramework>', "<TargetFramework>$TargetFramework</TargetFramework>"
            Set-Content -Path $ProjectPath -Value $content
            Write-Host "  ✅ Reverted to $TargetFramework" -ForegroundColor Green
            return $true
        } else {
            Write-Host "  ℹ️  Already targeting $TargetFramework or no change needed" -ForegroundColor Blue
            return $false
        }
    } else {
        Write-Host "  ❌ Project file not found: $ProjectPath" -ForegroundColor Red
        return $false
    }
}

# Function to fix escaped package references
function Fix-EscapedPackageReferences {
    param([string]$ProjectPath)
    
    if (Test-Path $ProjectPath) {
        $content = Get-Content $ProjectPath -Raw
        $updated = $false
        
        # Fix escaped package names that are causing issues
        $escapedPackages = @{
            'System\.Text\.Json' = 'System.Text.Json'
            'Microsoft\.Extensions\.Logging' = 'Microsoft.Extensions.Logging'
            'Microsoft\.Extensions\.DependencyInjection' = 'Microsoft.Extensions.DependencyInjection'
            'Microsoft\.Extensions\.Http' = 'Microsoft.Extensions.Http'
            'Microsoft\.Extensions\.Hosting' = 'Microsoft.Extensions.Hosting'
            'System\.Threading\.Channels' = 'System.Threading.Channels'
            'System\.ServiceProcess\.ServiceController' = 'System.ServiceProcess.ServiceController'
            'System\.Management' = 'System.Management'
        }
        
        foreach ($escaped in $escapedPackages.Keys) {
            $correct = $escapedPackages[$escaped]
            if ($content -match $escaped) {
                $content = $content -replace $escaped, $correct
                $updated = $true
                Write-Host "  🔧 Fixed escaped package reference: $escaped -> $correct" -ForegroundColor Green
            }
        }
        
        if ($updated) {
            Set-Content -Path $ProjectPath -Value $content
            Write-Host "  📦 Fixed escaped package references" -ForegroundColor Green
        }
    }
}

# Find all project files in the repository
Write-Host "🔍 Scanning for project files..." -ForegroundColor Yellow
$projectFiles = Get-ChildItem -Path "." -Recurse -Include "*.fsproj", "*.csproj" | Where-Object { $_.FullName -notmatch "\\bin\\|\\obj\\|\\node_modules\\" }

Write-Host "Found $($projectFiles.Count) project files" -ForegroundColor Green
Write-Host ""

# Track statistics
$totalProjects = 0
$revertedProjects = 0
$skippedProjects = 0

# Revert each project
foreach ($projectFile in $projectFiles) {
    $totalProjects++
    $relativePath = $projectFile.FullName.Replace((Get-Location).Path + "\", "")
    
    Write-Host "📁 Processing: $relativePath" -ForegroundColor Cyan
    
    $wasReverted = Revert-ProjectTargetFramework -ProjectPath $projectFile.FullName -TargetFramework "net9.0"
    Fix-EscapedPackageReferences -ProjectPath $projectFile.FullName
    
    if ($wasReverted) {
        $revertedProjects++
    } else {
        $skippedProjects++
    }
    
    Write-Host ""
}

# Summary
Write-Host "📊 REVERSION SUMMARY" -ForegroundColor Cyan
Write-Host "====================" -ForegroundColor Cyan
Write-Host "Total Projects Scanned: $totalProjects" -ForegroundColor White
Write-Host "Projects Reverted: $revertedProjects" -ForegroundColor Green
Write-Host "Projects Skipped (already .NET 9): $skippedProjects" -ForegroundColor Blue
Write-Host ""

Write-Host "🎯 Next Steps:" -ForegroundColor Yellow
Write-Host "1. Run 'dotnet restore Tars.sln' to restore packages" -ForegroundColor White
Write-Host "2. Run 'dotnet build Tars.sln -c Release' to test compilation" -ForegroundColor White
Write-Host "3. Run 'dotnet test Tars.sln -c Release' to run tests" -ForegroundColor White
Write-Host ""

Write-Host "✅ .NET 9 reversion completed!" -ForegroundColor Green
Write-Host ""
Write-Host "💡 Note: .NET 10 preview doesn't have all required packages available yet." -ForegroundColor Yellow
Write-Host "   We'll upgrade to .NET 10 when it's officially released with full package support." -ForegroundColor Yellow
