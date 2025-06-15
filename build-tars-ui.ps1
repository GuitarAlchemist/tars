#!/usr/bin/env pwsh
# TARS UI Build Script
# Builds the F# Fable UI project with all dependencies

param(
    [string]$Mode = "Development",
    [switch]$Clean,
    [switch]$Install,
    [switch]$Serve,
    [switch]$Production,
    [switch]$Help
)

if ($Help) {
    Write-Host "TARS UI Build Script" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage: ./build-tars-ui.ps1 [options]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Green
    Write-Host "  -Mode <Development|Production>  Build mode (default: Development)"
    Write-Host "  -Clean                          Clean build artifacts"
    Write-Host "  -Install                        Install dependencies"
    Write-Host "  -Serve                          Start development server"
    Write-Host "  -Production                     Build for production"
    Write-Host "  -Help                           Show this help"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Magenta
    Write-Host "  ./build-tars-ui.ps1 -Install   # Install all dependencies"
    Write-Host "  ./build-tars-ui.ps1 -Serve     # Start development server"
    Write-Host "  ./build-tars-ui.ps1 -Production # Build for production"
    exit 0
}

$ErrorActionPreference = "Stop"
$UIProjectPath = "TarsEngine.FSharp.UI"

Write-Host "🚀 TARS UI Build Script" -ForegroundColor Cyan
Write-Host "========================" -ForegroundColor Cyan

# Check prerequisites
function Test-Prerequisites {
    Write-Host "🔍 Checking prerequisites..." -ForegroundColor Yellow
    
    # Check .NET SDK
    try {
        $dotnetVersion = dotnet --version
        Write-Host "✅ .NET SDK: $dotnetVersion" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ .NET SDK not found. Please install .NET 8.0 SDK" -ForegroundColor Red
        exit 1
    }
    
    # Check Node.js
    try {
        $nodeVersion = node --version
        Write-Host "✅ Node.js: $nodeVersion" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Node.js not found. Please install Node.js 18+" -ForegroundColor Red
        exit 1
    }
    
    # Check npm
    try {
        $npmVersion = npm --version
        Write-Host "✅ npm: $npmVersion" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ npm not found. Please install npm" -ForegroundColor Red
        exit 1
    }
}

# Clean build artifacts
function Invoke-Clean {
    Write-Host "🧹 Cleaning build artifacts..." -ForegroundColor Yellow
    
    if (Test-Path "$UIProjectPath/bin") {
        Remove-Item "$UIProjectPath/bin" -Recurse -Force
        Write-Host "✅ Removed bin directory" -ForegroundColor Green
    }
    
    if (Test-Path "$UIProjectPath/obj") {
        Remove-Item "$UIProjectPath/obj" -Recurse -Force
        Write-Host "✅ Removed obj directory" -ForegroundColor Green
    }
    
    if (Test-Path "$UIProjectPath/dist") {
        Remove-Item "$UIProjectPath/dist" -Recurse -Force
        Write-Host "✅ Removed dist directory" -ForegroundColor Green
    }
    
    if (Test-Path "$UIProjectPath/node_modules") {
        Remove-Item "$UIProjectPath/node_modules" -Recurse -Force
        Write-Host "✅ Removed node_modules directory" -ForegroundColor Green
    }
}

# Install dependencies
function Install-Dependencies {
    Write-Host "📦 Installing dependencies..." -ForegroundColor Yellow
    
    # Install .NET dependencies
    Write-Host "Installing .NET packages..." -ForegroundColor Cyan
    Set-Location $UIProjectPath
    dotnet restore
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to restore .NET packages" -ForegroundColor Red
        exit 1
    }
    Write-Host "✅ .NET packages installed" -ForegroundColor Green
    
    # Install npm dependencies
    Write-Host "Installing npm packages..." -ForegroundColor Cyan
    npm install
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to install npm packages" -ForegroundColor Red
        exit 1
    }
    Write-Host "✅ npm packages installed" -ForegroundColor Green
    
    Set-Location ..
}

# Build F# project
function Build-FSharpProject {
    Write-Host "🔨 Building F# project..." -ForegroundColor Yellow
    
    Set-Location $UIProjectPath
    dotnet build --configuration $Mode
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to build F# project" -ForegroundColor Red
        exit 1
    }
    Write-Host "✅ F# project built successfully" -ForegroundColor Green
    Set-Location ..
}

# Build for production
function Build-Production {
    Write-Host "🏗️ Building for production..." -ForegroundColor Yellow
    
    Set-Location $UIProjectPath
    npm run build
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to build for production" -ForegroundColor Red
        exit 1
    }
    Write-Host "✅ Production build completed" -ForegroundColor Green
    Write-Host "📁 Output: $UIProjectPath/dist" -ForegroundColor Cyan
    Set-Location ..
}

# Start development server
function Start-DevServer {
    Write-Host "🌐 Starting development server..." -ForegroundColor Yellow
    
    Set-Location $UIProjectPath
    Write-Host "🚀 TARS UI will be available at http://localhost:3000" -ForegroundColor Cyan
    Write-Host "🔄 Hot reload enabled - changes will automatically refresh" -ForegroundColor Cyan
    Write-Host "⏹️ Press Ctrl+C to stop the server" -ForegroundColor Cyan
    Write-Host ""
    
    npm run dev
    Set-Location ..
}

# Check if UI project exists
if (-not (Test-Path $UIProjectPath)) {
    Write-Host "❌ UI project not found at $UIProjectPath" -ForegroundColor Red
    Write-Host "Please ensure you're running this script from the TARS root directory" -ForegroundColor Yellow
    exit 1
}

# Run prerequisite checks
Test-Prerequisites

# Handle command line options
if ($Clean) {
    Invoke-Clean
}

if ($Install) {
    Install-Dependencies
}

if ($Production) {
    $Mode = "Release"
    if (-not $Install) {
        Install-Dependencies
    }
    Build-FSharpProject
    Build-Production
    
    Write-Host ""
    Write-Host "🎉 Production build completed successfully!" -ForegroundColor Green
    Write-Host "📁 Files are ready in: $UIProjectPath/dist" -ForegroundColor Cyan
    Write-Host "🌐 Deploy these files to your web server" -ForegroundColor Cyan
}
elseif ($Serve) {
    if (-not $Install) {
        Install-Dependencies
    }
    Build-FSharpProject
    Start-DevServer
}
else {
    # Default build
    if (-not $Install) {
        Install-Dependencies
    }
    Build-FSharpProject
    
    Write-Host ""
    Write-Host "🎉 Build completed successfully!" -ForegroundColor Green
    Write-Host "🌐 Run './build-tars-ui.ps1 -Serve' to start the development server" -ForegroundColor Cyan
    Write-Host "🏗️ Run './build-tars-ui.ps1 -Production' to build for production" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "✨ TARS UI Build Complete ✨" -ForegroundColor Magenta
