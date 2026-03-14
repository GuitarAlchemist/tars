# TARS Development Environment Setup
# Automates the setup of TARS development environment

param(
    [switch]$SkipDotNet = $false,
    [switch]$SkipBuild = $false,
    [switch]$Verbose = $false
)

Write-Host "🛠️ TARS Development Environment Setup" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# Set verbose preference
if ($Verbose) {
    $VerbosePreference = "Continue"
}

function Test-DotNetInstallation {
    Write-Host "🔍 Checking .NET Installation..." -ForegroundColor Yellow
    
    try {
        $dotnetVersion = dotnet --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "   ✅ .NET SDK found: $dotnetVersion" -ForegroundColor Green
            return $true
        } else {
            Write-Host "   ❌ .NET SDK not found" -ForegroundColor Red
            return $false
        }
    } catch {
        Write-Host "   ❌ .NET SDK not available" -ForegroundColor Red
        return $false
    }
}

function Install-DotNet {
    Write-Host "📥 Installing .NET SDK..." -ForegroundColor Yellow
    
    try {
        # Download and install .NET SDK
        $installerUrl = "https://download.microsoft.com/download/8/8/5/88544F33-836A-49C5-8B67-451C24709A8F/dotnet-sdk-9.0.100-win-x64.exe"
        $installerPath = "$env:TEMP\dotnet-sdk-installer.exe"
        
        Write-Host "   Downloading .NET SDK installer..." -ForegroundColor Gray
        Invoke-WebRequest -Uri $installerUrl -OutFile $installerPath
        
        Write-Host "   Running .NET SDK installer..." -ForegroundColor Gray
        Start-Process -FilePath $installerPath -ArgumentList "/quiet" -Wait
        
        Write-Host "   ✅ .NET SDK installation completed" -ForegroundColor Green
        return $true
    } catch {
        Write-Host "   ❌ .NET SDK installation failed: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

function Test-ProjectStructure {
    Write-Host "🗂️ Verifying Project Structure..." -ForegroundColor Yellow
    
    $requiredDirectories = @(
        "TarsEngine.FSharp.Cli",
        "TarsEngine.FSharp.Metascripts",
        ".tars",
        ".tars/metascripts",
        ".tars/docs",
        ".tars/plans"
    )
    
    $missingDirectories = @()
    
    foreach ($dir in $requiredDirectories) {
        if (Test-Path $dir -PathType Container) {
            Write-Host "   ✅ $dir exists" -ForegroundColor Green
        } else {
            Write-Host "   ❌ $dir missing" -ForegroundColor Red
            $missingDirectories += $dir
        }
    }
    
    return $missingDirectories.Count -eq 0
}

function Restore-Dependencies {
    Write-Host "📦 Restoring Dependencies..." -ForegroundColor Yellow
    
    try {
        Write-Host "   Restoring CLI project dependencies..." -ForegroundColor Gray
        dotnet restore TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj
        
        Write-Host "   Restoring Metascripts project dependencies..." -ForegroundColor Gray
        dotnet restore TarsEngine.FSharp.Metascripts/TarsEngine.FSharp.Metascripts.fsproj
        
        Write-Host "   ✅ Dependencies restored successfully" -ForegroundColor Green
        return $true
    } catch {
        Write-Host "   ❌ Dependency restoration failed: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

function Build-Projects {
    Write-Host "🔨 Building Projects..." -ForegroundColor Yellow
    
    try {
        Write-Host "   Building Metascripts project..." -ForegroundColor Gray
        dotnet build TarsEngine.FSharp.Metascripts/TarsEngine.FSharp.Metascripts.fsproj
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "   ❌ Metascripts project build failed" -ForegroundColor Red
            return $false
        }
        
        Write-Host "   Building CLI project..." -ForegroundColor Gray
        dotnet build TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "   ❌ CLI project build failed" -ForegroundColor Red
            return $false
        }
        
        Write-Host "   ✅ All projects built successfully" -ForegroundColor Green
        return $true
    } catch {
        Write-Host "   ❌ Build failed: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

function Test-TarsInstallation {
    Write-Host "🧪 Testing TARS Installation..." -ForegroundColor Yellow
    
    try {
        Write-Host "   Testing version command..." -ForegroundColor Gray
        $versionOutput = dotnet run --project TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj -- version 2>&1
        
        if ($versionOutput -match "TARS") {
            Write-Host "   ✅ TARS CLI working correctly" -ForegroundColor Green
            
            Write-Host "   Testing metascript discovery..." -ForegroundColor Gray
            $metascriptOutput = dotnet run --project TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj -- metascript-list --discover 2>&1
            
            if ($metascriptOutput -match "metascripts") {
                $metascriptCount = ($metascriptOutput | Select-String "Loaded from").Count
                Write-Host "   ✅ Metascript engine working ($metascriptCount scripts discovered)" -ForegroundColor Green
                return $true
            } else {
                Write-Host "   ❌ Metascript engine not working" -ForegroundColor Red
                return $false
            }
        } else {
            Write-Host "   ❌ TARS CLI not working" -ForegroundColor Red
            return $false
        }
    } catch {
        Write-Host "   ❌ TARS testing failed: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

function Show-SetupSummary {
    Write-Host ""
    Write-Host "📋 Development Environment Setup Summary" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    
    Write-Host "✅ TARS development environment is ready!" -ForegroundColor Green
    Write-Host ""
    Write-Host "🎯 Available Commands:" -ForegroundColor Yellow
    Write-Host "   • dotnet run --project TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj -- help"
    Write-Host "   • dotnet run --project TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj -- metascript-list --discover"
    Write-Host "   • dotnet run --project TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj -- intelligence measure"
    Write-Host ""
    Write-Host "🚀 Next Steps:" -ForegroundColor Yellow
    Write-Host "   • Run demos: .\.tars\scripts\demo\tars-demo.ps1"
    Write-Host "   • Run tests: .\.tars\scripts\test\run-all-tests.ps1"
    Write-Host "   • Explore metascripts: .\.tars\scripts\demo\metascript-showcase.ps1"
    Write-Host ""
    Write-Host "📚 Documentation:" -ForegroundColor Yellow
    Write-Host "   • Configuration: .tars\tars.yaml"
    Write-Host "   • Plans: .tars\plans\README.md"
    Write-Host "   • Metascripts: .tars\metascripts\"
}

# Main setup process
Write-Host "🎯 Starting TARS Development Setup" -ForegroundColor Green
Write-Host ""

# Check .NET installation
if (-not $SkipDotNet) {
    if (-not (Test-DotNetInstallation)) {
        if (-not (Install-DotNet)) {
            Write-Host "❌ Setup failed: Could not install .NET SDK" -ForegroundColor Red
            exit 1
        }
    }
} else {
    Write-Host "⏭️ Skipping .NET installation check" -ForegroundColor Yellow
}

# Verify project structure
if (-not (Test-ProjectStructure)) {
    Write-Host "❌ Setup failed: Project structure is incomplete" -ForegroundColor Red
    Write-Host "Please ensure you're running this script from the TARS project root directory." -ForegroundColor Yellow
    exit 1
}

# Restore dependencies
if (-not (Restore-Dependencies)) {
    Write-Host "❌ Setup failed: Could not restore dependencies" -ForegroundColor Red
    exit 1
}

# Build projects
if (-not $SkipBuild) {
    if (-not (Build-Projects)) {
        Write-Host "❌ Setup failed: Could not build projects" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "⏭️ Skipping project build" -ForegroundColor Yellow
}

# Test installation
if (-not (Test-TarsInstallation)) {
    Write-Host "❌ Setup failed: TARS installation test failed" -ForegroundColor Red
    exit 1
}

Show-SetupSummary
