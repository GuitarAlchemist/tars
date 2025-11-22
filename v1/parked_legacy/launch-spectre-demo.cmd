@echo off
echo.
echo ========================================
echo   LAUNCHING TARS SPECTRE CONSOLE DEMO
echo ========================================
echo.
echo Opening new terminal window for spectacular display...
echo.

cd /d "%~dp0"

REM Try Windows Terminal first (modern)
where wt >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Using Windows Terminal for best experience...
    start wt -p "Command Prompt" cmd /k "cd /d \"%CD%\" && demo-spectre-widgets.cmd"
    goto :end
)

REM Fallback to regular command prompt in new window
echo Using Command Prompt in new window...
start "TARS Spectre Console Demo" cmd /k "cd /d \"%CD%\" && demo-spectre-widgets.cmd"

:end
echo.
echo Demo launched in new window!
echo Check the new terminal window for the spectacular Spectre.Console widgets demo.
echo.
pause
