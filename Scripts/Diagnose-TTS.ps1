# Diagnose-TTS.ps1
#
# This script diagnoses issues with the TTS installation
#
# Usage: .\Diagnose-TTS.ps1

Write-Host "TARS TTS Diagnostics" -ForegroundColor Cyan
Write-Host "===================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
try {
    $pythonVersion = python --version
    Write-Host "System Python version: $pythonVersion" -ForegroundColor Green
    
    # Extract version numbers
    $versionMatch = $pythonVersion -match '(\d+)\.(\d+)\.(\d+)'
    if ($versionMatch) {
        $majorVersion = [int]$Matches[1]
        $minorVersion = [int]$Matches[2]
        
        # Check if Python version is compatible with TTS
        if ($majorVersion -eq 3 -and $minorVersion -ge 9 -and $minorVersion -lt 12) {
            Write-Host "Python version is compatible with TTS" -ForegroundColor Green
        } else {
            Write-Host "Warning: Python version $majorVersion.$minorVersion is not fully compatible with TTS" -ForegroundColor Yellow
            Write-Host "TTS works best with Python 3.9, 3.10, or 3.11" -ForegroundColor Yellow
        }
    }
}
catch {
    Write-Host "Error: Python is not installed or not in PATH" -ForegroundColor Red
}

# Check for virtual environment
$venvPath = Join-Path $PSScriptRoot "tts-venv"
if (Test-Path $venvPath) {
    Write-Host "TTS virtual environment found at: $venvPath" -ForegroundColor Green
    
    # Check activation script
    $activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
    if (Test-Path $activateScript) {
        Write-Host "Virtual environment activation script found" -ForegroundColor Green
        
        # Activate the virtual environment
        Write-Host "Activating virtual environment..." -ForegroundColor Yellow
        & $activateScript
        
        # Check TTS installation in virtual environment
        try {
            $output = python -c "import TTS; print(f'TTS version: {TTS.__version__}')"
            Write-Host $output -ForegroundColor Green
            
            # Check if TTS models can be listed
            try {
                $output = python -c "from TTS.api import TTS; print(f'Available models: {len(TTS().list_models())}')"
                Write-Host $output -ForegroundColor Green
            }
            catch {
                Write-Host "Error listing TTS models: $_" -ForegroundColor Red
            }
        }
        catch {
            Write-Host "Error importing TTS in virtual environment: $_" -ForegroundColor Red
        }
        
        # Deactivate the virtual environment
        deactivate
    }
    else {
        Write-Host "Virtual environment activation script not found" -ForegroundColor Red
    }
}
else {
    Write-Host "TTS virtual environment not found" -ForegroundColor Red
    Write-Host "Run .\Install-TTS.ps1 to create the virtual environment" -ForegroundColor Yellow
}

# Check TTS server script
$pythonDir = Join-Path (Split-Path $PSScriptRoot -Parent) "Python"
$serverScriptPath = Join-Path $pythonDir "tts_server.py"
if (Test-Path $serverScriptPath) {
    Write-Host "TTS server script found at: $serverScriptPath" -ForegroundColor Green
    
    # Check server script content
    $scriptContent = Get-Content $serverScriptPath -Raw
    if ($scriptContent -match "from TTS.api import TTS") {
        Write-Host "TTS server script content looks valid" -ForegroundColor Green
    }
    else {
        Write-Host "TTS server script content may be invalid" -ForegroundColor Red
    }
}
else {
    Write-Host "TTS server script not found" -ForegroundColor Red
    Write-Host "The script will be created when TARS starts" -ForegroundColor Yellow
}

# Check if TTS server is running
try {
    $response = Invoke-WebRequest -Uri "http://localhost:5002/status" -Method GET -ErrorAction SilentlyContinue
    if ($response.StatusCode -eq 200) {
        Write-Host "TTS server is running" -ForegroundColor Green
        
        # Check server diagnostics
        try {
            $diagnostics = Invoke-WebRequest -Uri "http://localhost:5002/diagnostics" -Method GET -ErrorAction SilentlyContinue
            Write-Host "TTS server diagnostics:" -ForegroundColor Green
            Write-Host $diagnostics.Content -ForegroundColor Green
        }
        catch {
            Write-Host "Error getting TTS server diagnostics: $_" -ForegroundColor Red
        }
    }
    else {
        Write-Host "TTS server returned unexpected status code: $($response.StatusCode)" -ForegroundColor Red
    }
}
catch {
    Write-Host "TTS server is not running" -ForegroundColor Red
    Write-Host "Run .\run-tts-server.bat to start the server manually" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Diagnostics complete" -ForegroundColor Cyan
