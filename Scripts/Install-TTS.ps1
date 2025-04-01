# Install-TTS.ps1
#
# This script installs the TTS package and its dependencies for TARS
#
# Usage: .\Install-TTS.ps1

Write-Host "Installing TTS and dependencies for TARS..." -ForegroundColor Cyan

# Check Python version
try {
    $pythonVersion = python --version
    Write-Host "Python version: $pythonVersion" -ForegroundColor Green

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

            $installPython = Read-Host "Would you like to install Python 3.11? (y/n)"
            if ($installPython -eq "y") {
                # Download and install Python 3.11
                $pythonInstallerUrl = "https://www.python.org/ftp/python/3.11.8/python-3.11.8-amd64.exe"
                $pythonInstallerPath = "$env:TEMP\python-3.11.8-amd64.exe"

                Write-Host "Downloading Python 3.11..." -ForegroundColor Yellow
                Invoke-WebRequest -Uri $pythonInstallerUrl -OutFile $pythonInstallerPath

                Write-Host "Installing Python 3.11..." -ForegroundColor Yellow
                Start-Process -FilePath $pythonInstallerPath -ArgumentList "/quiet", "InstallAllUsers=1", "PrependPath=1" -Wait

                Write-Host "Python 3.11 installed. Please restart your terminal and run this script again." -ForegroundColor Green
                exit 0
            }
        }
    }
}
catch {
    Write-Host "Error: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.11 from https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install dependencies first
Write-Host "Installing dependencies..." -ForegroundColor Yellow
python -m pip install flask soundfile numpy langdetect

# Extract Python version for TTS compatibility
$majorVersion = [int]$Matches[1]
$minorVersion = [int]$Matches[2]

# Select appropriate TTS version based on Python version
$ttsVersion = "0.17.6"
if ($majorVersion -eq 3) {
    if ($minorVersion -eq 9 -or $minorVersion -eq 10 -or $minorVersion -eq 11) {
        $ttsVersion = "0.17.6"
    } elseif ($minorVersion -eq 8) {
        $ttsVersion = "0.14.3"
    } elseif ($minorVersion -eq 7) {
        $ttsVersion = "0.14.3"
    } else {
        Write-Host "Warning: Using latest TTS version, but it may not be compatible with Python $majorVersion.$minorVersion" -ForegroundColor Yellow
    }
}

# Install TTS with specific version
Write-Host "Installing TTS package version $ttsVersion..." -ForegroundColor Yellow

# Create a virtual environment for TTS
$venvPath = Join-Path $PSScriptRoot "tts-venv"
if (-not (Test-Path $venvPath)) {
    Write-Host "Creating virtual environment for TTS..." -ForegroundColor Yellow
    python -m venv $venvPath
}

# Activate the virtual environment
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & $activateScript

    # Upgrade pip in the virtual environment
    Write-Host "Upgrading pip in virtual environment..." -ForegroundColor Yellow
    python -m pip install --upgrade pip

    # Install dependencies first
    Write-Host "Installing dependencies in virtual environment..." -ForegroundColor Yellow
    python -m pip install flask soundfile numpy langdetect

    # Try to install TTS
    Write-Host "Installing TTS version $ttsVersion in virtual environment..." -ForegroundColor Yellow
    python -m pip install TTS==$ttsVersion

    # Check if installation was successful
    $ttsInstalled = $false
    try {
        $output = python -c "import TTS; print('TTS installed successfully')"
        if ($output -like "*TTS installed successfully*") {
            $ttsInstalled = $true
        }
    } catch {
        $ttsInstalled = $false
    }

    if ($ttsInstalled) {
        Write-Host "TTS installation completed successfully!" -ForegroundColor Green

        # Create a test script
        $testScriptPath = Join-Path $PSScriptRoot "test-tts.py"
        $testScript = @"
from TTS.api import TTS
import sys
import traceback

print(f"Python version: {sys.version}")
print("TTS package imported successfully")

try:
    # List available models
    models = TTS().list_models()
    print(f"Available models: {len(models)}")
    for i, model in enumerate(models[:5]):
        print(f"  {i+1}. {model}")
    if len(models) > 5:
        print(f"  ... and {len(models) - 5} more")

    # Try to load a model
    model_name = "tts_models/en/ljspeech/tacotron2-DDC"
    print(f"Loading model: {model_name}")
    tts = TTS(model_name=model_name, progress_bar=True)
    print("Model loaded successfully")

    # Try to generate speech
    print("Generating speech...")
    wav = tts.tts("Hello, this is a test of the TARS text-to-speech system.")
    print("Speech generated successfully")

    # Try to save to file
    import io
    import soundfile as sf
    print("Saving to file...")
    sf.write("test_tts_output.wav", wav, 22050, format='WAV')
    print("Speech saved to test_tts_output.wav")

except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()
"@
        Set-Content -Path $testScriptPath -Value $testScript

        Write-Host "Created test script at $testScriptPath" -ForegroundColor Green
        Write-Host "You can run it with: python $testScriptPath" -ForegroundColor Green

        # Create a batch file to activate the environment and run TARS
        $batchFilePath = Join-Path $PSScriptRoot "run-tars-with-tts.bat"
        $batchFileContent = @"
@echo off
echo Activating TTS virtual environment...
call "$venvPath\Scripts\activate.bat"
echo Running TARS...
cd ..
tarscli %*
"@
        Set-Content -Path $batchFilePath -Value $batchFileContent

        Write-Host "Created batch file at $batchFilePath" -ForegroundColor Green
        Write-Host "You can run TARS with TTS support using: $batchFilePath" -ForegroundColor Green

        # Deactivate the virtual environment
        deactivate
    } else {
        Write-Host "TTS installation failed. Trying alternative installation method..." -ForegroundColor Red

        # Try alternative installation
        python -m pip install TTS==$ttsVersion --no-deps
        python -m pip install numpy==1.24.3 torch==2.0.1 librosa==0.10.1

        # Check again
        try {
            $output = python -c "import TTS; print('TTS installed successfully')"
            if ($output -like "*TTS installed successfully*") {
                Write-Host "TTS installation completed successfully with alternative method!" -ForegroundColor Green
            } else {
                Write-Host "TTS installation failed. Please try installing manually:" -ForegroundColor Red
                Write-Host "python -m pip install TTS==$ttsVersion" -ForegroundColor Yellow
            }
        } catch {
            Write-Host "TTS installation failed. Please try installing manually:" -ForegroundColor Red
            Write-Host "python -m pip install TTS==$ttsVersion" -ForegroundColor Yellow
        }

        # Deactivate the virtual environment
        deactivate
    }
} else {
    Write-Host "Error: Could not find activation script for virtual environment" -ForegroundColor Red
}

Write-Host "Done!" -ForegroundColor Cyan
