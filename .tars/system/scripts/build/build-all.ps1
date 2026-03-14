# TARS Build Script
# Builds all TARS projects with comprehensive options

param(
    [string]$Configuration = "Debug",
    [switch]$Clean = $false,
    [switch]$Restore = $true,
    [switch]$Test = $false,
    [switch]$Package = $false,
    [switch]$Verbose = $false
)

Write-Host "🔨 TARS Build Script" -ForegroundColor Cyan
Write-Host "===================" -ForegroundColor Cyan
Write-Host ""

# Set verbose preference
if ($Verbose) {
    $VerbosePreference = "Continue"
}

$BuildResults = @{
    Success = $true
    Projects = @()
    StartTime = Get-Date
}

function Add-BuildResult {
    param(
        [string]$ProjectName,
        [bool]$Success,
        [string]$Message = ""
    )
    
    $BuildResults.Projects += @{
        Name = $ProjectName
        Success = $Success
        Message = $Message
        Timestamp = Get-Date
    }
    
    if (-not $Success) {
        $BuildResults.Success = $false
    }
}

function Clean-Projects {
    Write-Host "🧹 Cleaning Projects..." -ForegroundColor Yellow
    
    try {
        Write-Host "   Cleaning Metascripts project..." -ForegroundColor Gray
        dotnet clean TarsEngine.FSharp.Metascripts/TarsEngine.FSharp.Metascripts.fsproj --configuration $Configuration
        
        Write-Host "   Cleaning CLI project..." -ForegroundColor Gray
        dotnet clean TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj --configuration $Configuration
        
        Write-Host "   ✅ Clean completed successfully" -ForegroundColor Green
        return $true
    } catch {
        Write-Host "   ❌ Clean failed: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

function Restore-Projects {
    Write-Host "📦 Restoring Projects..." -ForegroundColor Yellow
    
    try {
        Write-Host "   Restoring Metascripts project..." -ForegroundColor Gray
        dotnet restore TarsEngine.FSharp.Metascripts/TarsEngine.FSharp.Metascripts.fsproj
        
        if ($LASTEXITCODE -ne 0) {
            Add-BuildResult -ProjectName "TarsEngine.FSharp.Metascripts" -Success $false -Message "Restore failed"
            return $false
        }
        
        Write-Host "   Restoring CLI project..." -ForegroundColor Gray
        dotnet restore TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj
        
        if ($LASTEXITCODE -ne 0) {
            Add-BuildResult -ProjectName "TarsEngine.FSharp.Cli" -Success $false -Message "Restore failed"
            return $false
        }
        
        Write-Host "   ✅ Restore completed successfully" -ForegroundColor Green
        return $true
    } catch {
        Write-Host "   ❌ Restore failed: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

function Build-Project {
    param(
        [string]$ProjectPath,
        [string]$ProjectName
    )
    
    Write-Host "   Building $ProjectName..." -ForegroundColor Gray
    
    try {
        $buildArgs = @(
            "build",
            $ProjectPath,
            "--configuration", $Configuration,
            "--no-restore"
        )
        
        if ($Verbose) {
            $buildArgs += "--verbosity", "detailed"
        }
        
        & dotnet @buildArgs
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "   ✅ $ProjectName build successful" -ForegroundColor Green
            Add-BuildResult -ProjectName $ProjectName -Success $true -Message "Build successful"
            return $true
        } else {
            Write-Host "   ❌ $ProjectName build failed" -ForegroundColor Red
            Add-BuildResult -ProjectName $ProjectName -Success $false -Message "Build failed with exit code $LASTEXITCODE"
            return $false
        }
    } catch {
        Write-Host "   ❌ $ProjectName build exception: $($_.Exception.Message)" -ForegroundColor Red
        Add-BuildResult -ProjectName $ProjectName -Success $false -Message $_.Exception.Message
        return $false
    }
}

function Build-AllProjects {
    Write-Host "🔨 Building All Projects ($Configuration)..." -ForegroundColor Yellow
    
    $projects = @(
        @{ Path = "TarsEngine.FSharp.Metascripts/TarsEngine.FSharp.Metascripts.fsproj"; Name = "TarsEngine.FSharp.Metascripts" },
        @{ Path = "TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj"; Name = "TarsEngine.FSharp.Cli" }
    )
    
    $allSuccess = $true
    
    foreach ($project in $projects) {
        if (-not (Build-Project -ProjectPath $project.Path -ProjectName $project.Name)) {
            $allSuccess = $false
        }
    }
    
    return $allSuccess
}

function Test-Projects {
    Write-Host "🧪 Running Tests..." -ForegroundColor Yellow
    
    try {
        # Run our comprehensive test script
        $testScript = ".\.tars\scripts\test\run-all-tests.ps1"
        
        if (Test-Path $testScript) {
            Write-Host "   Running TARS test suite..." -ForegroundColor Gray
            & $testScript -SkipBuild -TestCategory "all"
            
            if ($LASTEXITCODE -eq 0) {
                Write-Host "   ✅ All tests passed" -ForegroundColor Green
                return $true
            } else {
                Write-Host "   ❌ Some tests failed" -ForegroundColor Red
                return $false
            }
        } else {
            Write-Host "   ⚠️ Test script not found, skipping tests" -ForegroundColor Yellow
            return $true
        }
    } catch {
        Write-Host "   ❌ Test execution failed: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

function Package-Projects {
    Write-Host "📦 Packaging Projects..." -ForegroundColor Yellow
    
    try {
        $outputDir = ".\build\packages"
        
        if (-not (Test-Path $outputDir)) {
            New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
        }
        
        Write-Host "   Publishing CLI project..." -ForegroundColor Gray
        dotnet publish TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj `
            --configuration $Configuration `
            --output "$outputDir\cli" `
            --self-contained false
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "   ✅ CLI packaging successful" -ForegroundColor Green
            return $true
        } else {
            Write-Host "   ❌ CLI packaging failed" -ForegroundColor Red
            return $false
        }
    } catch {
        Write-Host "   ❌ Packaging failed: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

function Show-BuildSummary {
    $endTime = Get-Date
    $duration = $endTime - $BuildResults.StartTime
    
    Write-Host ""
    Write-Host "📊 Build Summary" -ForegroundColor Cyan
    Write-Host "================" -ForegroundColor Cyan
    Write-Host ""
    
    Write-Host "Configuration: $Configuration" -ForegroundColor White
    Write-Host "Duration: $($duration.TotalSeconds.ToString('F1')) seconds" -ForegroundColor White
    Write-Host ""
    
    Write-Host "Project Results:" -ForegroundColor White
    foreach ($project in $BuildResults.Projects) {
        $status = if ($project.Success) { "✅ PASSED" } else { "❌ FAILED" }
        $color = if ($project.Success) { "Green" } else { "Red" }
        
        Write-Host "   $($project.Name): $status" -ForegroundColor $color
        if (-not $project.Success -and $project.Message) {
            Write-Host "      $($project.Message)" -ForegroundColor Red
        }
    }
    
    Write-Host ""
    
    if ($BuildResults.Success) {
        Write-Host "🎉 Build completed successfully!" -ForegroundColor Green
        Write-Host ""
        Write-Host "🚀 Next Steps:" -ForegroundColor Yellow
        Write-Host "   • Run TARS: dotnet run --project TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj -- help"
        Write-Host "   • Run demos: .\.tars\scripts\demo\tars-demo.ps1"
        Write-Host "   • Run tests: .\.tars\scripts\test\run-all-tests.ps1"
    } else {
        Write-Host "❌ Build failed. Please review the errors above." -ForegroundColor Red
    }
}

# Main build process
Write-Host "🎯 Starting TARS Build Process" -ForegroundColor Green
Write-Host "Configuration: $Configuration" -ForegroundColor White
Write-Host ""

# Clean if requested
if ($Clean) {
    if (-not (Clean-Projects)) {
        Write-Host "❌ Build failed during clean step" -ForegroundColor Red
        exit 1
    }
}

# Restore if requested
if ($Restore) {
    if (-not (Restore-Projects)) {
        Write-Host "❌ Build failed during restore step" -ForegroundColor Red
        exit 1
    }
}

# Build all projects
if (-not (Build-AllProjects)) {
    Write-Host "❌ Build failed during compilation" -ForegroundColor Red
    Show-BuildSummary
    exit 1
}

# Run tests if requested
if ($Test) {
    if (-not (Test-Projects)) {
        Write-Host "⚠️ Build succeeded but tests failed" -ForegroundColor Yellow
        Show-BuildSummary
        exit 1
    }
}

# Package if requested
if ($Package) {
    if (-not (Package-Projects)) {
        Write-Host "⚠️ Build succeeded but packaging failed" -ForegroundColor Yellow
        Show-BuildSummary
        exit 1
    }
}

Show-BuildSummary

# Exit with appropriate code
exit $(if ($BuildResults.Success) { 0 } else { 1 })
