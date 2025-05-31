@echo off
echo.
echo ========================================
echo   TARS Swarm Interactive Demo
echo ========================================
echo.

cd /d "%~dp0"

echo Building TARS Swarm Demo...
dotnet build TarsSwarmDemo/TarsSwarmDemo.fsproj --verbosity quiet

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ Build failed! Please check the errors above.
    pause
    exit /b 1
)

echo.
echo ✅ Build successful! Starting interactive demo...
echo.
echo 🎯 Use arrow keys to navigate the menu
echo 📊 Explore different TARS swarm features
echo 🚪 Select "Exit Demo" when finished
echo.

dotnet run --project TarsSwarmDemo/TarsSwarmDemo.fsproj -- interactive

echo.
echo Demo completed. Press any key to exit...
pause > nul
