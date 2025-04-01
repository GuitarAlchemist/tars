@echo off
REM TARS Simple Demo Batch File
REM This script demonstrates basic TARS capabilities with commands known to work

echo.
echo ===================================================
echo             TARS SIMPLE DEMONSTRATION
echo ===================================================
echo.
echo This batch file will demonstrate the key features of TARS.
echo Press Ctrl+C at any time to exit the demo.
echo.

REM Check if TARS CLI exists
if not exist "%~dp0..\TarsCli\bin\Debug\net9.0\tarscli.exe" (
    echo Error: TARS CLI not found.
    echo Please build the TARS CLI project first by running 'dotnet build' in the TarsCli directory.
    goto :EOF
)

set TARS_CLI="%~dp0..\TarsCli\bin\Debug\net9.0\tarscli.exe"
set CURRENT_DIR=%CD%
cd "%~dp0.."

REM Prompt for demo topic
set DEMO_TOPIC=Artificial Intelligence
echo Default demo topic: %DEMO_TOPIC%
echo.
set /p CUSTOM_TOPIC="Enter a custom topic for the demo (or press Enter to use the default): "
if not "%CUSTOM_TOPIC%"=="" set DEMO_TOPIC=%CUSTOM_TOPIC%
echo Using topic: %DEMO_TOPIC%
echo.

echo Starting demo in 3 seconds...
timeout /t 3 > nul

REM Section 1: Basic Information
echo.
echo ===================================================
echo                 BASIC INFORMATION
echo ===================================================
echo.

echo TARS Version
echo -----------
echo.
echo ^> tarscli --version
%TARS_CLI% --version
echo.

REM Section 2: Chat Bot
echo.
echo ===================================================
echo                    CHAT BOT
echo ===================================================
echo.

echo Single Message Chat
echo ------------------
echo.
echo ^> tarscli chat --message "Explain %DEMO_TOPIC% in 3 sentences" --model llama3
%TARS_CLI% chat --message "Explain %DEMO_TOPIC% in 3 sentences" --model llama3
echo.

timeout /t 3 > nul

REM Section 3: Console Capture
echo.
echo ===================================================
echo                CONSOLE CAPTURE
echo ===================================================
echo.

echo Console Capture Demo
echo ------------------
echo.
echo ^> tarscli console-capture --start
%TARS_CLI% console-capture --start
echo.

echo Generating some console output...
echo Warning: This is a test warning message
echo Error: This is a test error message
echo Info: This is a test info message
echo Debug: This is a test debug message
echo.

echo ^> tarscli console-capture --stop
%TARS_CLI% console-capture --stop
echo.

timeout /t 3 > nul

REM Section 4: Demo Mode
echo.
echo ===================================================
echo                   DEMO MODE
echo ===================================================
echo.

echo Running TARS Demo
echo ----------------
echo.
echo ^> tarscli demo run --interactive
%TARS_CLI% demo run --interactive
echo.

REM Conclusion
echo.
echo ===================================================
echo                 DEMO COMPLETE
echo ===================================================
echo.
echo Thank you for exploring TARS capabilities!
echo.
echo For more information, visit:
echo https://github.com/GuitarAlchemist/tars
echo.
echo To run specific commands, use:
echo %TARS_CLI% [command] [options]
echo.

REM Return to the original directory
cd "%CURRENT_DIR%"

pause
