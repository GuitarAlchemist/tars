#!/usr/bin/env pwsh

# TARS .NET 9 Migration Script
# Updates all projects to target .NET 9 for consistency

Write-Host "🚀 TARS .NET 9 Migration Script" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""

# Function to update project file target framework
function Update-ProjectTargetFramework {
    param(
        [string]$ProjectPath,
        [string]$TargetFramework = "net9.0"
    )
    
    if (Test-Path $ProjectPath) {
        Write-Host "📝 Updating $ProjectPath..." -ForegroundColor Yellow
        
        $content = Get-Content $ProjectPath -Raw
        
        # Update various target framework patterns
        $patterns = @(
            '<TargetFramework>net6\.0</TargetFramework>',
            '<TargetFramework>net8\.0</TargetFramework>',
            '<TargetFramework>net7\.0</TargetFramework>'
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

# Function to update package references for .NET 9 compatibility
function Update-PackageReferences {
    param([string]$ProjectPath)
    
    if (Test-Path $ProjectPath) {
        $content = Get-Content $ProjectPath -Raw
        $updated = $false
        
        # Update FSharp.Core to compatible version
        if ($content -match '<PackageReference Include="FSharp\.Core" Version="8\.0\.400"') {
            $content = $content -replace '<PackageReference Include="FSharp\.Core" Version="8\.0\.400"', '<PackageReference Include="FSharp.Core" Version="9.0.300"'
            $updated = $true
        }
        
        # Update System.Text.Json to secure version
        if ($content -match '<PackageReference Include="System\.Text\.Json" Version="8\.0\.0"') {
            $content = $content -replace '<PackageReference Include="System\.Text\.Json" Version="8\.0\.0"', '<PackageReference Include="System.Text.Json" Version="8.0.5"'
            $updated = $true
        }
        
        # Add .NET 9 specific properties if needed
        if ($content -notmatch '<SuppressNETCoreSdkPreviewMessage>') {
            $content = $content -replace '(<PropertyGroup>)', "`$1`n    <SuppressNETCoreSdkPreviewMessage>true</SuppressNETCoreSdkPreviewMessage>"
            $updated = $true
        }
        
        if ($updated) {
            Set-Content -Path $ProjectPath -Value $content
            Write-Host "  📦 Updated package references" -ForegroundColor Green
        }
    }
}

# List of all TARS projects to update
$projectsToUpdate = @(
    # Core F# Projects
    "TarsEngine.FSharp.Core/TarsEngine.FSharp.Core.fsproj",
    "TarsEngine.FSharp.Core.Tests/TarsEngine.FSharp.Core.Tests.fsproj",
    "TarsEngine.FSharp.Main/TarsEngine.FSharp.Main.fsproj",
    
    # UI Projects
    "TarsEngine.FSharp.UI/TarsEngine.FSharp.UI.fsproj",
    "TarsEngine.FSharp.DynamicUI/TarsEngine.FSharp.DynamicUI.fsproj",
    
    # Agent Projects
    "TarsEngine.FSharp.Agents/TarsEngine.FSharp.Agents.fsproj",
    
    # DSL Projects
    "TarsEngine.DSL/TarsEngine.DSL.fsproj",
    "TarsEngine.DSL.Tests/TarsEngine.DSL.Tests.fsproj",
    
    # ML Projects
    "TarsEngine.FSharp.ML/TarsEngine.FSharp.ML.fsproj",
    
    # Legacy Projects
    "TarsEngine.FSharp/TarsEngine.FSharp.fsproj",
    "TarsEngine.TreeOfThought/TarsEngine.TreeOfThought.fsproj",
    "TarsEngineFSharp.Core/TarsEngineFSharp.Core.fsproj",
    
    # Additional Projects (if they exist)
    "TarsEngine.FSharp.Metascript/TarsEngine.FSharp.Metascript.fsproj",
    "TarsEngine.FSharp.Metascript.Runner/TarsEngine.FSharp.Metascript.Runner.fsproj",
    "TarsEngine.FSharp.DataSources/TarsEngine.FSharp.DataSources.fsproj",
    "TarsEngine.FSharp.Testing/TarsEngine.FSharp.Testing.fsproj",
    "TarsEngine.FSharp.Packaging/TarsEngine.FSharp.Packaging.fsproj",
    "TarsEngine.FSharp.ProjectGeneration/TarsEngine.FSharp.ProjectGeneration.fsproj",
    "TarsEngine.FSharp.Consciousness/TarsEngine.FSharp.Consciousness.fsproj"
)

# Track statistics
$totalProjects = 0
$updatedProjects = 0
$skippedProjects = 0
$notFoundProjects = 0

Write-Host "🔍 Scanning for TARS projects..." -ForegroundColor Yellow
Write-Host ""

# Update each project
foreach ($project in $projectsToUpdate) {
    $totalProjects++
    
    if (Test-Path $project) {
        Write-Host "📁 Processing: $project" -ForegroundColor Cyan
        
        $wasUpdated = Update-ProjectTargetFramework -ProjectPath $project -TargetFramework "net9.0"
        Update-PackageReferences -ProjectPath $project
        
        if ($wasUpdated) {
            $updatedProjects++
        } else {
            $skippedProjects++
        }
        
        Write-Host ""
    } else {
        Write-Host "⚠️  Project not found: $project" -ForegroundColor DarkYellow
        $notFoundProjects++
    }
}

# Update Directory.Build.props for global .NET 9 settings
Write-Host "🌐 Updating Directory.Build.props..." -ForegroundColor Cyan

$directoryBuildProps = @"
<Project>
    <PropertyGroup>
        <!-- Global .NET 9 Settings -->
        <TargetFramework>net9.0</TargetFramework>
        <FSharpCoreVersion>9.0.300</FSharpCoreVersion>
        <SuppressNETCoreSdkPreviewMessage>true</SuppressNETCoreSdkPreviewMessage>
        <TreatWarningsAsErrors>false</TreatWarningsAsErrors>
        <NoWarn>NU1608;NU1903;NU1701</NoWarn>
        
        <!-- TARS Specific Settings -->
        <Authors>TARS Development Team</Authors>
        <Company>TARS Project</Company>
        <Product>TARS - Autonomous Reasoning System</Product>
        <Version>3.0.0</Version>
        <LangVersion>9.0</LangVersion>
    </PropertyGroup>

    <ItemGroup>
        <!-- Global Package References -->
        <PackageReference Update="FSharp.Core" Version="`$(FSharpCoreVersion)" />
        <PackageReference Update="System.Text.Json" Version="8.0.5" />
        <PackageReference Update="Microsoft.Extensions.Logging" Version="9.0.0" />
        <PackageReference Update="Microsoft.Extensions.Logging.Console" Version="9.0.0" />
    </ItemGroup>
</Project>
"@

Set-Content -Path "Directory.Build.props" -Value $directoryBuildProps
Write-Host "✅ Created/Updated Directory.Build.props with .NET 9 settings" -ForegroundColor Green
Write-Host ""

# Create global.json for .NET 9 SDK
Write-Host "🌐 Creating global.json..." -ForegroundColor Cyan

$globalJson = @"
{
  "sdk": {
    "version": "9.0.100",
    "rollForward": "latestMajor",
    "allowPrerelease": true
  },
  "msbuild-sdks": {
    "Microsoft.Build.NoTargets": "3.7.0"
  }
}
"@

Set-Content -Path "global.json" -Value $globalJson
Write-Host "✅ Created global.json for .NET 9 SDK" -ForegroundColor Green
Write-Host ""

# Summary
Write-Host "📊 MIGRATION SUMMARY" -ForegroundColor Cyan
Write-Host "===================" -ForegroundColor Cyan
Write-Host "Total Projects Scanned: $totalProjects" -ForegroundColor White
Write-Host "Projects Updated: $updatedProjects" -ForegroundColor Green
Write-Host "Projects Skipped (already .NET 9): $skippedProjects" -ForegroundColor Blue
Write-Host "Projects Not Found: $notFoundProjects" -ForegroundColor Yellow
Write-Host ""

if ($updatedProjects -gt 0) {
    Write-Host "🎉 Successfully migrated $updatedProjects projects to .NET 9!" -ForegroundColor Green
} else {
    Write-Host "ℹ️  All projects were already targeting .NET 9 or compatible versions." -ForegroundColor Blue
}

Write-Host ""
Write-Host "🔧 NEXT STEPS:" -ForegroundColor Yellow
Write-Host "1. Run 'dotnet restore' to restore packages with .NET 9" -ForegroundColor White
Write-Host "2. Run 'dotnet build' to verify all projects compile" -ForegroundColor White
Write-Host "3. Run tests to ensure functionality is preserved" -ForegroundColor White
Write-Host "4. Update CI/CD pipelines to use .NET 9 SDK" -ForegroundColor White
Write-Host ""

Write-Host "✅ .NET 9 migration completed!" -ForegroundColor Green
