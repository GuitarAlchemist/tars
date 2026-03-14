@echo off
REM TARS Code Protection Demo Runner (Windows Batch)
REM Simple wrapper for PowerShell script

echo 🛡️ TARS CODE PROTECTION DEMO
echo ============================
echo.

REM Check if PowerShell is available
where pwsh >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    where powershell >nul 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo ❌ PowerShell not found. Please install PowerShell.
        pause
        exit /b 1
    )
    set PS_CMD=powershell
) else (
    set PS_CMD=pwsh
)

REM Run the PowerShell script with arguments
%PS_CMD% -ExecutionPolicy Bypass -File "run-tars-protection-demo.ps1" %*

echo.
echo Press any key to continue...
pause >nul
