@echo off
REM TARS Capabilities Demo Batch File
REM This script demonstrates TARS capabilities by running various TARS CLI commands

echo.
echo ===================================================
echo             TARS CAPABILITIES DEMONSTRATION
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

echo TARS Version and Help
echo ---------------------
echo.
echo ^> tarscli --version
%TARS_CLI% --version
echo.
echo ^> tarscli help
%TARS_CLI% help
echo.

timeout /t 3 > nul

REM Section 2: Deep Thinking
echo.
echo ===================================================
echo                  DEEP THINKING
echo ===================================================
echo.

echo Run Deep Thinking Demo
echo ---------------------
echo.
echo ^> tarscli demo --type deep-thinking
%TARS_CLI% demo --type deep-thinking
echo.
echo.

timeout /t 3 > nul

REM Section 3: Chat Bot
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

REM Section 4: Speech System
echo.
echo ===================================================
echo                  SPEECH SYSTEM
echo ===================================================
echo.

echo Available Voices
echo ---------------
echo.
echo ^> tarscli speech list-voices
%TARS_CLI% speech list-voices
echo.

echo Text-to-Speech Demo
echo ------------------
echo.
echo ^> tarscli speech speak --text "Hello, I am TARS. I can help you with %DEMO_TOPIC%." --language en
%TARS_CLI% speech speak --text "Hello, I am TARS. I can help you with %DEMO_TOPIC%." --language en
echo.

timeout /t 3 > nul

REM Section 5: Console Capture
echo.
echo ===================================================
echo                CONSOLE CAPTURE
echo ===================================================
echo.

echo Start Capturing Console Output
echo ----------------------------
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

echo Stop Capturing Console Output
echo ---------------------------
echo.
echo ^> tarscli console-capture --stop
%TARS_CLI% console-capture --stop
echo.

timeout /t 3 > nul

REM Section 6: MCP (Model Context Protocol)
echo.
echo ===================================================
echo         MODEL CONTEXT PROTOCOL (MCP)
echo ===================================================
echo.

echo MCP Status
echo ----------
echo.
echo ^> tarscli mcp status
%TARS_CLI% mcp status
echo.

timeout /t 3 > nul

REM Section 7: Self-Improvement
echo.
echo ===================================================
echo               SELF-IMPROVEMENT
echo ===================================================
echo.

echo Self-Improvement Status
echo ----------------------
echo.
echo ^> tarscli auto-improve --status
%TARS_CLI% auto-improve --status
echo.

timeout /t 3 > nul

REM Section 8: Documentation
echo.
echo ===================================================
echo                 DOCUMENTATION
echo ===================================================
echo.

echo List Documentation
echo -----------------
echo.
echo ^> tarscli docs list
%TARS_CLI% docs list
echo.

timeout /t 3 > nul

REM Section 9: Language Specification
echo.
echo ===================================================
echo            LANGUAGE SPECIFICATION
echo ===================================================
echo.

echo TARS DSL Specification
echo ---------------------
echo.
echo ^> tarscli lang spec --dsl --preview
%TARS_CLI% lang spec --dsl --preview
echo.

timeout /t 3 > nul

REM Section 10: Demo Mode
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
