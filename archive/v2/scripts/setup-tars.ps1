#Requires -RunAsAdministrator
<#
.SYNOPSIS
    TARS v2 - Automated Setup Script for Windows
.DESCRIPTION
    This script installs all dependencies required to run TARS on a fresh Windows machine.
    It supports both Winget and Chocolatey package managers.
.EXAMPLE
    .\setup-tars.ps1
    .\setup-tars.ps1 -SkipModels
    .\setup-tars.ps1 -UseChocolatey
.NOTES
    Author: TARS Team
    Version: 1.0.0
    Requires: Windows 10/11, PowerShell 5.1+, Administrator privileges
#>

param(
    [switch]$SkipModels,        # Skip downloading LLM models
    [switch]$UseChocolatey,     # Use Chocolatey instead of Winget
    [switch]$SkipDocker,        # Skip Docker installation
    [switch]$SkipLlamaCpp,      # Skip llama.cpp installation
    [switch]$CpuOnly,           # Use CPU-only binaries (no CUDA)
    [string]$InstallPath = "C:\tools"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# Colors for output
function Write-Step { param($msg) Write-Host "`n🔧 $msg" -ForegroundColor Cyan }
function Write-Success { param($msg) Write-Host "✅ $msg" -ForegroundColor Green }
function Write-Warn { param($msg) Write-Host "⚠️ $msg" -ForegroundColor Yellow }
function Write-Error { param($msg) Write-Host "❌ $msg" -ForegroundColor Red }

Write-Host @"

╔════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║     ████████╗ █████╗ ██████╗ ███████╗    ██╗   ██╗██████╗           ║
║     ╚══██╔══╝██╔══██╗██╔══██╗██╔════╝    ██║   ██║╚════██╗          ║
║        ██║   ███████║██████╔╝███████╗    ██║   ██║ █████╔╝          ║
║        ██║   ██╔══██║██╔══██╗╚════██║    ╚██╗ ██╔╝██╔═══╝           ║
║        ██║   ██║  ██║██║  ██║███████║     ╚████╔╝ ███████╗          ║
║        ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝      ╚═══╝  ╚══════╝          ║
║                                                                      ║
║                    Autonomous Reasoning System                       ║
║                         Setup Script v1.0                            ║
╚════════════════════════════════════════════════════════════════════╝

"@ -ForegroundColor Magenta

# Check system requirements
Write-Step "Checking system requirements..."

$os = Get-CimInstance Win32_OperatingSystem
if ($os.Caption -notlike "*Windows 10*" -and $os.Caption -notlike "*Windows 11*") {
    Write-Error "This script requires Windows 10 or Windows 11"
    exit 1
}

$ram = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
if ($ram -lt 8) {
    Write-Warn "You have ${ram}GB RAM. 16GB+ recommended for larger models."
}

# Check for NVIDIA GPU
$gpu = Get-CimInstance Win32_VideoController | Where-Object { $_.Name -like "*NVIDIA*" }
if ($gpu) {
    Write-Success "NVIDIA GPU detected: $($gpu.Name)"
    $hasNvidia = $true
}
else {
    Write-Warn "No NVIDIA GPU detected. Using CPU-only mode."
    $hasNvidia = $false
    $CpuOnly = $true
}

Write-Success "System check passed (OS: $($os.Caption), RAM: ${ram}GB)"

# Function to check if a command exists
function Test-Command {
    param($Command)
    return [bool](Get-Command $Command -ErrorAction SilentlyContinue)
}

# Function to install via Winget
function Install-Winget {
    param($PackageId, $Name)
    Write-Step "Installing $Name via Winget..."
    try {
        winget install --id $PackageId --accept-source-agreements --accept-package-agreements --silent
        Write-Success "$Name installed"
    }
    catch {
        Write-Warn "Failed to install $Name via Winget: $_"
    }
}

# Function to install via Chocolatey
function Install-Choco {
    param($Package, $Name)
    Write-Step "Installing $Name via Chocolatey..."
    try {
        choco install $Package -y
        Write-Success "$Name installed"
    }
    catch {
        Write-Warn "Failed to install $Name via Chocolatey: $_"
    }
}

# Install package manager if needed
if ($UseChocolatey) {
    if (-not (Test-Command "choco")) {
        Write-Step "Installing Chocolatey..."
        Set-ExecutionPolicy Bypass -Scope Process -Force
        [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
        iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
        Write-Success "Chocolatey installed"
    }
}
else {
    if (-not (Test-Command "winget")) {
        Write-Warn "Winget not found. Please install 'App Installer' from Microsoft Store."
        Write-Warn "Alternatively, run with -UseChocolatey flag."
        exit 1
    }
}

# Install .NET SDK
if (-not (Test-Command "dotnet") -or -not ((dotnet --list-sdks) -match "10\.")) {
    if ($UseChocolatey) {
        Install-Choco "dotnet-sdk" ".NET SDK"
    }
    else {
        Install-Winget "Microsoft.DotNet.SDK.10" ".NET 10 SDK"
    }
}
else {
    Write-Success ".NET SDK already installed"
}

# Install Git
if (-not (Test-Command "git")) {
    if ($UseChocolatey) {
        Install-Choco "git" "Git"
    }
    else {
        Install-Winget "Git.Git" "Git"
    }
}
else {
    Write-Success "Git already installed"
}

# Install CUDA (if NVIDIA GPU)
if ($hasNvidia -and -not $CpuOnly) {
    $cudaInstalled = Test-Path "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if (-not $cudaInstalled) {
        if ($UseChocolatey) {
            Install-Choco "cuda" "NVIDIA CUDA"
        }
        else {
            Install-Winget "Nvidia.CUDA" "NVIDIA CUDA"
        }
    }
    else {
        Write-Success "CUDA already installed"
    }
}

# Install Ollama
if (-not (Test-Command "ollama")) {
    if ($UseChocolatey) {
        Install-Choco "ollama" "Ollama"
    }
    else {
        Install-Winget "Ollama.Ollama" "Ollama"
    }
}
else {
    Write-Success "Ollama already installed"
}

# Install Docker (optional)
if (-not $SkipDocker) {
    if (-not (Test-Command "docker")) {
        if ($UseChocolatey) {
            Install-Choco "docker-desktop" "Docker Desktop"
        }
        else {
            Install-Winget "Docker.DockerDesktop" "Docker Desktop"
        }
    }
    else {
        Write-Success "Docker already installed"
    }
}

# Install llama.cpp
if (-not $SkipLlamaCpp) {
    $llamaPath = "$InstallPath\llama-cpp"
    
    if (-not (Test-Path "$llamaPath\llama-server.exe")) {
        Write-Step "Installing llama.cpp..."
        
        New-Item -ItemType Directory -Path $llamaPath -Force | Out-Null
        
        try {
            $release = Invoke-RestMethod "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
            
            if ($CpuOnly) {
                $assetPattern = "*win-cpu-x64*"
            }
            else {
                $assetPattern = "*win-cuda-12.4*"
            }
            
            $asset = $release.assets | Where-Object { $_.name -like $assetPattern -and $_.name -notlike "*cudart*" } | Select-Object -First 1
            
            Write-Host "  Downloading llama.cpp ($($asset.name))..."
            Invoke-WebRequest -Uri $asset.browser_download_url -OutFile "$env:TEMP\llama.zip"
            Expand-Archive "$env:TEMP\llama.zip" -DestinationPath $llamaPath -Force
            
            # Download CUDA runtime if needed
            if (-not $CpuOnly) {
                $cudart = $release.assets | Where-Object { $_.name -like "*cudart*cuda-12.4*" } | Select-Object -First 1
                if ($cudart) {
                    Write-Host "  Downloading CUDA runtime..."
                    Invoke-WebRequest -Uri $cudart.browser_download_url -OutFile "$env:TEMP\cudart.zip"
                    Expand-Archive "$env:TEMP\cudart.zip" -DestinationPath $llamaPath -Force
                }
            }
            
            # Add to PATH
            $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
            if ($userPath -notlike "*$llamaPath*") {
                [Environment]::SetEnvironmentVariable("Path", "$userPath;$llamaPath", "User")
                $env:Path += ";$llamaPath"
            }
            
            Write-Success "llama.cpp installed to $llamaPath"
        }
        catch {
            Write-Warn "Failed to install llama.cpp: $_"
        }
    }
    else {
        Write-Success "llama.cpp already installed"
    }
}

# Create models directory
$modelsPath = "C:\models"
New-Item -ItemType Directory -Path $modelsPath -Force | Out-Null

# Download models
if (-not $SkipModels) {
    Write-Step "Downloading LLM models..."
    
    # Start Ollama service
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 3
    
    # Pull recommended models
    $models = @(
        "qwen3:14b",         # Best general thinking model
        "nomic-embed-text"   # Embedding model
    )
    
    foreach ($model in $models) {
        Write-Host "  Pulling $model..."
        try {
            ollama pull $model
            Write-Success "$model downloaded"
        }
        catch {
            Write-Warn "Failed to pull $model: $_"
        }
    }
    
    # Download GGUF for llama.cpp
    $ggufPath = "$modelsPath\Qwen3-8B-Q4_K_M.gguf"
    if (-not (Test-Path $ggufPath)) {
        Write-Host "  Downloading Qwen3-8B GGUF model (~5GB)..."
        try {
            curl.exe -L -o $ggufPath "https://huggingface.co/unsloth/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf"
            Write-Success "Qwen3-8B GGUF downloaded"
        }
        catch {
            Write-Warn "Failed to download GGUF model: $_"
        }
    }
}

# Clone TARS if not in repo
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$tarsPath = Split-Path -Parent $scriptPath

if (-not (Test-Path "$tarsPath\Tars.sln")) {
    Write-Step "Cloning TARS repository..."
    $tarsPath = "$env:USERPROFILE\source\repos\tars\v2"
    git clone https://github.com/GuitarAlchemist/tars.git "$env:USERPROFILE\source\repos\tars"
    Set-Location $tarsPath
}

# Build TARS
Write-Step "Building TARS..."
Set-Location $tarsPath
dotnet restore
dotnet build

if ($LASTEXITCODE -eq 0) {
    Write-Success "TARS built successfully"
}
else {
    Write-Error "Build failed"
    exit 1
}

# Create startup scripts
Write-Step "Creating startup scripts..."

$startLlamaScript = @"
@echo off
echo Starting llama.cpp server...
C:\tools\llama-cpp\llama-server.exe -m "C:\models\Qwen3-8B-Q4_K_M.gguf" --host 127.0.0.1 --port 8080 -c 4096 -ngl 99
pause
"@
$startLlamaScript | Out-File -FilePath "$tarsPath\start-llama.bat" -Encoding ASCII

$startTarsScript = @"
@echo off
echo Starting TARS UI...
cd /d "$tarsPath"
dotnet run --project src/Tars.Interface.Ui/Tars.Interface.Ui.fsproj
pause
"@
$startTarsScript | Out-File -FilePath "$tarsPath\start-tars.bat" -Encoding ASCII

$startAllScript = @"
@echo off
echo Starting TARS with llama.cpp backend...
start "llama.cpp" cmd /c "$tarsPath\start-llama.bat"
timeout /t 5 /nobreak
start "TARS UI" cmd /c "$tarsPath\start-tars.bat"
timeout /t 3 /nobreak
start http://localhost:5000
"@
$startAllScript | Out-File -FilePath "$tarsPath\start-all.bat" -Encoding ASCII

Write-Success "Startup scripts created"

# Final summary
Write-Host @"

╔════════════════════════════════════════════════════════════════════╗
║                     🎉 SETUP COMPLETE! 🎉                           ║
╚════════════════════════════════════════════════════════════════════╝

"@ -ForegroundColor Green

Write-Host "Installed Components:" -ForegroundColor Cyan
Write-Host "  ✅ .NET SDK"
Write-Host "  ✅ Git"
Write-Host "  ✅ Ollama"
if (-not $SkipLlamaCpp) { Write-Host "  ✅ llama.cpp ($(if($CpuOnly){'CPU'}else{'CUDA'}))" }
if (-not $SkipDocker) { Write-Host "  ✅ Docker Desktop" }
if (-not $SkipModels) { Write-Host "  ✅ LLM Models (qwen3, nomic-embed-text)" }
Write-Host "  ✅ TARS v2"

Write-Host "`nQuick Start:" -ForegroundColor Cyan
Write-Host "  1. Run: .\start-all.bat"
Write-Host "  2. Open: http://localhost:5000"

Write-Host "`nManual Start:" -ForegroundColor Cyan
Write-Host "  Terminal 1: .\start-llama.bat"
Write-Host "  Terminal 2: .\start-tars.bat"

Write-Host "`nPaths:" -ForegroundColor Cyan
Write-Host "  TARS:      $tarsPath"
Write-Host "  llama.cpp: $InstallPath\llama-cpp"
Write-Host "  Models:    $modelsPath"

Write-Host "`n" -ForegroundColor White
