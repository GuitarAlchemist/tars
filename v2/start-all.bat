@echo off
echo.
echo ╔════════════════════════════════════════════════════════════╗
echo ║                                                              ║
echo ║     ████████╗ █████╗ ██████╗ ███████╗    ██╗   ██╗██████╗   ║
echo ║     ╚══██╔══╝██╔══██╗██╔══██╗██╔════╝    ██║   ██║╚════██╗  ║
echo ║        ██║   ███████║██████╔╝███████╗    ██║   ██║ █████╔╝  ║
echo ║        ██║   ██╔══██║██╔══██╗╚════██║    ╚██╗ ██╔╝██╔═══╝   ║
echo ║        ██║   ██║  ██║██║  ██║███████║     ╚████╔╝ ███████╗  ║
echo ║        ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝      ╚═══╝  ╚══════╝  ║
echo ║                                                              ║
echo ║               Autonomous Reasoning System                    ║
echo ╚════════════════════════════════════════════════════════════╝
echo.
echo Starting all services...
echo.

:: Start llama.cpp server in background
echo [1/3] Starting llama.cpp server...
start "llama.cpp" cmd /c "%~dp0start-llama.bat"

:: Wait for llama.cpp to initialize
echo [2/3] Waiting for llama.cpp to initialize...
timeout /t 8 /nobreak > nul

:: Start TARS UI
echo [3/3] Starting TARS UI...
start "TARS UI" cmd /c "%~dp0start-tars.bat"

:: Wait a moment then open browser
timeout /t 5 /nobreak > nul
echo.
echo Opening browser to http://localhost:5000 ...
start http://localhost:5000

echo.
echo ═══════════════════════════════════════════════════════════════
echo   TARS is running!
echo   - UI: http://localhost:5000
echo   - llama.cpp: http://localhost:8080
echo   - Close this window to see running services
echo ═══════════════════════════════════════════════════════════════
echo.
pause
