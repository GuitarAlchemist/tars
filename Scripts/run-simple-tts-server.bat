@echo off
echo Starting simple TTS server...

set SCRIPT_DIR=%~dp0
set PYTHON_DIR=%SCRIPT_DIR%..\Python
set SERVER_SCRIPT=%PYTHON_DIR%\tts_server_simple.py

if not exist "%PYTHON_DIR%" mkdir "%PYTHON_DIR%"

echo Installing required packages...
python -m pip install flask gtts

echo Starting TTS server...
python "%SERVER_SCRIPT%"

echo TTS server stopped.
