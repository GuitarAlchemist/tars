# Speak-Text.ps1
#
# This script uses the Google Text-to-Speech API to speak text
#
# Usage: .\Speak-Text.ps1 -Text "Hello, world" -Language "en"

param(
    [Parameter(Mandatory=$true)]
    [string]$Text,

    [Parameter(Mandatory=$false)]
    [string]$Language = "en",

    [Parameter(Mandatory=$false)]
    [switch]$ShowDetails
)

# Enable verbose output if requested
if ($ShowDetails) {
    $VerbosePreference = "Continue"
    $DebugPreference = "Continue"
}

# Find Python executable
function Find-Python {
    $pythonCommands = @("python", "python3", "py")

    foreach ($cmd in $pythonCommands) {
        try {
            $version = & $cmd --version 2>&1
            if ($version -match "Python") {
                Write-Verbose "Found Python: $version using command '$cmd'"
                return $cmd
            }
        } catch {
            # Command not found, try next
        }
    }

    # Check common installation paths
    $commonPaths = @(
        "C:\Python312\python.exe",
        "C:\Python311\python.exe",
        "C:\Python310\python.exe",
        "C:\Python39\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe"
    )

    foreach ($path in $commonPaths) {
        if (Test-Path $path) {
            Write-Verbose "Found Python at path: $path"
            return $path
        }
    }

    Write-Error "Python not found. Please install Python 3.9 or later."
    return $null
}

$pythonCmd = Find-Python
if (-not $pythonCmd) {
    exit 1
}

# Install required packages if not already installed
Write-Verbose "Checking for gtts package..."
try {
    & $pythonCmd -c "import gtts" 2>$null
    Write-Verbose "gtts package is already installed"
} catch {
    Write-Host "Installing gtts module..." -ForegroundColor Yellow
    & $pythonCmd -m pip install gtts
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to install gtts package. Error code: $LASTEXITCODE"
        exit 1
    }
}

# Create a temporary Python script
$tempScript = [System.IO.Path]::GetTempFileName() + ".py"

# Escape special characters in the text
$escapedText = $Text -replace '"', '\"' -replace "\r\n|\n", " "

$pythonCode = @"
from gtts import gTTS
import sys
import os
import tempfile
import platform
import time

try:
    # Get text and language from command line arguments
    text = "$escapedText"
    lang = "$Language"

    print(f"Generating speech for: {text[:50]}{'...' if len(text) > 50 else ''}")
    print(f"Language: {lang}")

    # Create gTTS object
    tts = gTTS(text=text, lang=lang, slow=False)

    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
    temp_file.close()
    tts.save(temp_file.name)
    print(f"Speech saved to: {temp_file.name}")

    # Play the audio
    if platform.system() == 'Windows':
        os.system(f'start {temp_file.name}')
    elif platform.system() == 'Darwin':  # macOS
        os.system(f'afplay {temp_file.name}')
    else:  # Linux
        os.system(f'xdg-open {temp_file.name}')

    print("Speech playback started")

    # Keep the script running for a moment to ensure playback starts
    time.sleep(1)

    # Don't delete the file immediately so it can be played
    # The OS will clean it up eventually

    sys.exit(0)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
"@

# Write the Python script to the temporary file
Set-Content -Path $tempScript -Value $pythonCode

# Run the Python script
Write-Host "Generating speech..." -ForegroundColor Cyan
try {
    & $pythonCmd $tempScript
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to generate speech. Error code: $LASTEXITCODE"
        exit 1
    }
} catch {
    Write-Error "Exception while generating speech: $_"
    exit 1
} finally {
    # Clean up
    if (Test-Path $tempScript) {
        Remove-Item -Path $tempScript -ErrorAction SilentlyContinue
    }
}

Write-Host "Speech generation completed successfully" -ForegroundColor Green
exit 0
