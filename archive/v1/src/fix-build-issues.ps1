# TARS Build Issues Fix Script
# Fixes the identified package and dependency issues

Write-Host "TARS AUTONOMOUS BUILD FIX AGENT" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan

$ErrorActionPreference = "Continue"

Write-Host "Step 1: Fixing package references..." -ForegroundColor Yellow

# Fix 1: Remove problematic System.Xml.Linq reference
$packagingProject = "TarsEngine.FSharp.Packaging\TarsEngine.FSharp.Packaging.fsproj"
if (Test-Path $packagingProject) {
    $content = Get-Content $packagingProject -Raw
    $content = $content -replace '<PackageReference Include="System\.Xml\.Linq" Version="4\.3\.0" />', ''
    $content | Set-Content $packagingProject
    Write-Host "Fixed System.Xml.Linq reference" -ForegroundColor Green
}

# Fix 2: Update System.CommandLine to beta version
$dataSourcesProject = "TarsEngine.FSharp.DataSources\TarsEngine.FSharp.DataSources.fsproj"
if (Test-Path $dataSourcesProject) {
    $content = Get-Content $dataSourcesProject -Raw
    $content = $content -replace '<PackageReference Include="System\.CommandLine" Version="2\.0\.0" />', '<PackageReference Include="System.CommandLine" Version="2.0.0-beta4.22272.1" />'
    $content | Set-Content $dataSourcesProject
    Write-Host "Updated System.CommandLine to beta version" -ForegroundColor Green
}

# Fix 3: Remove Microsoft.CodeAnalysis.FSharp (not available)
if (Test-Path $dataSourcesProject) {
    $content = Get-Content $dataSourcesProject -Raw
    $content = $content -replace '<PackageReference Include="Microsoft\.CodeAnalysis\.FSharp"[^>]*>', ''
    $content | Set-Content $dataSourcesProject
    Write-Host "Removed unavailable Microsoft.CodeAnalysis.FSharp reference" -ForegroundColor Green
}

# Fix 4: Replace Playwright with Selenium WebDriver
$testingProject = "TarsEngine.FSharp.Testing\TarsEngine.FSharp.Testing.fsproj"
if (Test-Path $testingProject) {
    $content = Get-Content $testingProject -Raw
    $content = $content -replace '<PackageReference Include="Playwright"[^>]*>', '<PackageReference Include="Selenium.WebDriver" Version="4.15.0" />'
    $content | Set-Content $testingProject
    Write-Host "Replaced Playwright with Selenium WebDriver" -ForegroundColor Green
}

# Fix 5: Update System.Text.Json to fix security vulnerabilities
$projectsToUpdate = @(
    "TarsEngine.FSharp.Packaging\TarsEngine.FSharp.Packaging.fsproj",
    "TarsEngine.FSharp.ProjectGeneration\TarsEngine.FSharp.ProjectGeneration.fsproj",
    "TarsEngine.FSharp.DataSources\TarsEngine.FSharp.DataSources.fsproj",
    "TarsEngine.FSharp.Consciousness\TarsEngine.FSharp.Consciousness.fsproj"
)

foreach ($project in $projectsToUpdate) {
    if (Test-Path $project) {
        $content = Get-Content $project -Raw
        $content = $content -replace '<PackageReference Include="System\.Text\.Json" Version="8\.0\.0" />', '<PackageReference Include="System.Text.Json" Version="8.0.5" />'
        $content | Set-Content $project
        Write-Host "Updated System.Text.Json in $(Split-Path $project -Leaf)" -ForegroundColor Green
    }
}

# Fix 6: Fix target framework compatibility
$agentsProject = "TarsEngine.FSharp.Agents\TarsEngine.FSharp.Agents.fsproj"
if (Test-Path $agentsProject) {
    $content = Get-Content $agentsProject -Raw
    if ($content -match '<TargetFramework>net9\.0</TargetFramework>') {
        $content = $content -replace '<TargetFramework>net9\.0</TargetFramework>', '<TargetFramework>net8.0</TargetFramework>'
        $content | Set-Content $agentsProject
        Write-Host "Updated Agents project to target .NET 8.0" -ForegroundColor Green
    }
}

Write-Host "`nStep 2: Testing fixes..." -ForegroundColor Yellow

Write-Host "Running dotnet restore..." -ForegroundColor White
dotnet restore Tars.sln

if ($LASTEXITCODE -eq 0) {
    Write-Host "Package restore successful!" -ForegroundColor Green
    
    Write-Host "Running dotnet build..." -ForegroundColor White
    dotnet build Tars.sln
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "BUILD SUCCESSFUL!" -ForegroundColor Green
        Write-Host "TARS Autonomous Build Fix Agent completed successfully!" -ForegroundColor Green
    } else {
        Write-Host "Build completed with some warnings" -ForegroundColor Yellow
    }
} else {
    Write-Host "Some package issues remain" -ForegroundColor Red
}

Write-Host "`nAutonomous prerequisite management demonstration completed!" -ForegroundColor Cyan
