#Requires -Version 5.0

# Script to install Ollama and required models for TARS
param(
    [switch]$Force,
    [switch]$SkipOllama,
    [string[]]$Models = @("llama3.2", "all-minilm")
)

$ErrorActionPreference = "Stop"
$ollamaPath = "$env:ProgramFiles\Ollama\ollama.exe"

function Write-Status($message) {
    Write-Host ">> $message" -ForegroundColor Cyan
}

function Test-OllamaInstalled {
    return (Test-Path $ollamaPath) -or (Get-Command ollama -ErrorAction SilentlyContinue)
}

function Install-Ollama {
    Write-Status "Installing Ollama..."
    
    $tempDir = [System.IO.Path]::GetTempPath()
    $installerPath = Join-Path $tempDir "ollama-installer.exe"
    
    try {
        # Download Ollama installer
        $downloadUrl = "https://github.com/ollama/ollama/releases/latest/download/ollama-windows-amd64.exe"
        Write-Status "Downloading Ollama from $downloadUrl"
        Invoke-WebRequest -Uri $downloadUrl -OutFile $installerPath
        
        # Run installer
        Write-Status "Running Ollama installer..."
        Start-Process -FilePath $installerPath -ArgumentList "/S" -Wait
        
        # Verify installation
        if (Test-OllamaInstalled) {
            Write-Status "Ollama installed successfully!"
        } else {
            throw "Ollama installation failed. Please install manually from https://ollama.com"
        }
    }
    catch {
        Write-Host "Error installing Ollama: $_" -ForegroundColor Red
        Write-Host "Please install Ollama manually from https://ollama.com" -ForegroundColor Yellow
        exit 1
    }
    finally {
        if (Test-Path $installerPath) {
            Remove-Item $installerPath -Force
        }
    }
}

function Install-Models {
    param (
        [string[]]$ModelList
    )
    
    Write-Status "Installing required models..."
    
    foreach ($model in $ModelList) {
        Write-Status "Pulling model: $model"
        try {
            & ollama pull $model
            if ($LASTEXITCODE -ne 0) {
                Write-Host "Warning: Failed to pull model $model" -ForegroundColor Yellow
            } else {
                Write-Host "Model $model installed successfully" -ForegroundColor Green
            }
        }
        catch {
            Write-Host "Error pulling model $model: $_" -ForegroundColor Red
        }
    }
}

function Test-OllamaRunning {
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -ErrorAction SilentlyContinue
        return $true
    }
    catch {
        return $false
    }
}

function Start-OllamaService {
    Write-Status "Starting Ollama service..."
    
    try {
        Start-Process -FilePath $ollamaPath -WindowStyle Hidden
        
        # Wait for service to start
        $attempts = 0
        $maxAttempts = 10
        
        while (-not (Test-OllamaRunning) -and $attempts -lt $maxAttempts) {
            Write-Host "Waiting for Ollama service to start..." -ForegroundColor Yellow
            Start-Sleep -Seconds 2
            $attempts++
        }
        
        if (Test-OllamaRunning) {
            Write-Status "Ollama service started successfully!"
        } else {
            Write-Host "Warning: Ollama service did not start in the expected time." -ForegroundColor Yellow
            Write-Host "You may need to start it manually." -ForegroundColor Yellow
        }
    }
    catch {
        Write-Host "Error starting Ollama service: $_" -ForegroundColor Red
        Write-Host "Please start Ollama manually." -ForegroundColor Yellow
    }
}

# Main script execution
Write-Host "TARS Prerequisites Installer" -ForegroundColor Green
Write-Host "==========================" -ForegroundColor Green

# Check if Ollama is already installed
if (-not (Test-OllamaInstalled)) {
    if (-not $SkipOllama) {
        Install-Ollama
    } else {
        Write-Host "Ollama is not installed but installation was skipped." -ForegroundColor Yellow
        Write-Host "Please install Ollama manually from https://ollama.com" -ForegroundColor Yellow
    }
} else {
    Write-Status "Ollama is already installed."
}

# Check if Ollama service is running
if (-not (Test-OllamaRunning)) {
    Start-OllamaService
}

# Install required models
Install-Models -ModelList $Models

Write-Host "`nSetup completed!" -ForegroundColor Green
Write-Host "You can now use TARS with Ollama and the installed models." -ForegroundColor Green