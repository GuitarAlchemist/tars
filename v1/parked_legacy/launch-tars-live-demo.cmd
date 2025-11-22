@echo off
echo.
echo ========================================
echo   🚀 LAUNCHING TARS MIXTRAL MoE LIVE DEMO
echo ========================================
echo.
echo Opening spectacular live demo in new window...
echo This will showcase:
echo   * Real-time Mixtral MoE processing
echo   * Live expert routing and selection
echo   * Spectacular Spectre.Console widgets
echo   * Progress bars, tables, and panels
echo   * Actual AI analysis results
echo.

cd /d "%~dp0"

REM Build first to ensure everything is ready
echo Building TARS CLI...
dotnet build TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj --verbosity quiet

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ Build failed! Please check the errors above.
    pause
    exit /b 1
)

echo ✅ Build successful!
echo.

REM Try Windows Terminal first (best experience)
where wt >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo 🚀 Launching in Windows Terminal for best experience...
    start wt -p "Command Prompt" --title "TARS Mixtral MoE Live Demo" cmd /k "cd /d \"%CD%\" && dotnet run --project TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj -- mixtral live"
    goto :success
)

REM Fallback to PowerShell (good colors)
where powershell >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo 🚀 Launching in PowerShell window...
    start powershell -NoExit -Command "cd '%CD%'; dotnet run --project TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj -- mixtral live"
    goto :success
)

REM Final fallback to Command Prompt
echo 🚀 Launching in Command Prompt window...
start "TARS Mixtral MoE Live Demo" cmd /k "cd /d \"%CD%\" && dotnet run --project TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj -- mixtral live"

:success
echo.
echo ========================================
echo   ✅ DEMO LAUNCHED SUCCESSFULLY!
echo ========================================
echo.
echo The spectacular TARS Mixtral MoE live demo is now running
echo in a new terminal window with:
echo.
echo 🎯 FEATURES SHOWCASED:
echo   ✓ Live Mixtral MoE processing
echo   ✓ Real-time expert routing
echo   ✓ Spectacular Spectre.Console widgets
echo   ✓ Progress bars with live updates
echo   ✓ Expert status tables
echo   ✓ AI analysis results panels
echo   ✓ Figlet text headers
echo   ✓ Color-coded status displays
echo.
echo 📊 DATA SOURCES PROCESSED:
echo   ✓ GitHub Trending repositories
echo   ✓ Hacker News top stories
echo   ✓ Cryptocurrency markets
echo   ✓ Stack Overflow questions
echo   ✓ Reddit technology discussions
echo.
echo 🧠 EXPERTS DEMONSTRATED:
echo   ✓ CodeGeneration Expert
echo   ✓ CodeAnalysis Expert
echo   ✓ Architecture Expert
echo   ✓ Testing Expert
echo   ✓ Security Expert
echo   ✓ Performance Expert
echo.
echo Check the new terminal window to see the spectacular
echo live processing in action!
echo.
echo Press any key to exit this launcher...
pause > nul
