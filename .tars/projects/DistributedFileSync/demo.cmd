@echo off
:: Simple launcher for TARS Distributed File Sync Demo
:: Developed by TARS Multi-Agent Development Team

echo.
echo [96m🚀 TARS DISTRIBUTED FILE SYNC DEMO LAUNCHER[0m
echo [96m===========================================[0m
echo.
echo [92m👥 Developed by TARS Multi-Agent Team[0m
echo [97m   🏗️ Architect • 💻 Developer • 🔬 Researcher • ⚡ Optimizer[0m
echo [97m   🛡️ Security • 🤝 Coordinator • 🧪 QA Engineer[0m
echo.

:: Check if demo files exist
if exist "run-demo.cmd" (
    echo [92m✅ Windows Batch Demo: run-demo.cmd[0m
) else (
    echo [91m❌ Windows Batch Demo: run-demo.cmd not found[0m
)

if exist "run-demo.ps1" (
    echo [92m✅ PowerShell Demo: run-demo.ps1[0m
) else (
    echo [91m❌ PowerShell Demo: run-demo.ps1 not found[0m
)

echo.
echo [93mChoose your preferred demo launcher:[0m
echo.
echo [97m1. Windows Batch (.cmd) - Simple and compatible[0m
echo [97m2. PowerShell (.ps1) - Advanced features and colors[0m
echo [97m3. Manual instructions[0m
echo [97m4. Exit[0m
echo.
set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" goto run_batch
if "%choice%"=="2" goto run_powershell
if "%choice%"=="3" goto manual
if "%choice%"=="4" goto exit
echo [91mInvalid choice. Please enter 1, 2, 3, or 4.[0m
pause
goto start

:run_batch
echo.
echo [96m🚀 Launching Windows Batch Demo...[0m
call run-demo.cmd
goto end

:run_powershell
echo.
echo [96m🚀 Launching PowerShell Demo...[0m
powershell -ExecutionPolicy Bypass -File "run-demo.ps1"
goto end

:manual
echo.
echo [96m📋 MANUAL INSTRUCTIONS[0m
echo [96m=====================[0m
echo.
echo [97m🔧 Prerequisites:[0m
echo    [93m• .NET 9.0 SDK[0m
echo    [93m• Visual Studio or VS Code (optional)[0m
echo.
echo [97m🚀 Quick Start:[0m
echo    [93m1. dotnet restore[0m
echo    [93m2. dotnet build[0m
echo    [93m3. cd src\DistributedFileSync.Api[0m
echo    [93m4. dotnet run[0m
echo.
echo [97m🌐 Access Points:[0m
echo    [93m• API: https://localhost:5001[0m
echo    [93m• Swagger UI: https://localhost:5001[0m
echo    [93m• Health: https://localhost:5001/api/filesync/health[0m
echo.
echo [97m📁 Key Files:[0m
echo    [93m• README.md - Complete documentation[0m
echo    [93m• DistributedFileSync.sln - Visual Studio solution[0m
echo    [93m• src\ - Source code directory[0m
echo.
goto end

:exit
echo.
echo [92mThank you for exploring TARS Multi-Agent Development![0m
goto end

:end
echo.
echo [96m🎉 TARS Multi-Agent Team - Autonomous Software Development[0m
echo.
pause
