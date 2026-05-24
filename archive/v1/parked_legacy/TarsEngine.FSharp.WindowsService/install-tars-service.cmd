@echo off
REM TARS Windows Service Installation
REM Quick installer for TARS autonomous development engine

echo.
echo 🤖 TARS Windows Service Quick Installer
echo ═══════════════════════════════════════
echo.

REM Check for Administrator privileges
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ❌ This installer requires Administrator privileges.
    echo Right-click this file and select "Run as administrator"
    echo.
    pause
    exit /b 1
)

echo ✅ Running with Administrator privileges
echo.

REM Build the project first
echo 🔨 Building TARS Windows Service...
dotnet build TarsEngine.FSharp.WindowsService.fsproj --configuration Debug
if %errorLevel% neq 0 (
    echo ❌ Build failed. Please check for compilation errors.
    pause
    exit /b 1
)
echo ✅ Build completed successfully
echo.

REM Run the PowerShell installation script
echo 🚀 Installing TARS Windows Service...
powershell -ExecutionPolicy Bypass -File "install-service.ps1"

echo.
echo 🎉 Installation process completed!
echo.
pause
