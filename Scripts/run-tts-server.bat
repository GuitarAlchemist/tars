@echo off
echo Starting TTS server with virtual environment...

set SCRIPT_DIR=%~dp0
set VENV_PATH=%SCRIPT_DIR%tts-venv
set PYTHON_DIR=%SCRIPT_DIR%..\Python
set SERVER_SCRIPT=%PYTHON_DIR%\tts_server.py

if not exist "%PYTHON_DIR%" mkdir "%PYTHON_DIR%"

if not exist "%VENV_PATH%" (
    echo Virtual environment not found. Please run Install-TTS.ps1 first.
    exit /b 1
)

if not exist "%SERVER_SCRIPT%" (
    echo TTS server script not found. It will be created when TARS starts.
    exit /b 1
)

echo Activating virtual environment...
call "%VENV_PATH%\Scripts\activate.bat"

echo Starting TTS server...
python "%SERVER_SCRIPT%"

echo TTS server stopped.
deactivate
