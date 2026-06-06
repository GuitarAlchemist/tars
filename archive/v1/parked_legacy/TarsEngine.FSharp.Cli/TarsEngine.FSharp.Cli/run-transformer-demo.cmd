@echo off
echo.
echo ========================================
echo   TARS Transformer Demo Launcher
echo ========================================
echo.
echo Choose a demo:
echo   1. Model Recommendations (default)
echo   2. Performance Metrics
echo   3. Exit
echo.
set /p choice="Enter your choice (1-3): "

if "%choice%"=="2" (
    echo.
    echo Starting Performance Metrics Demo...
    start cmd /k "dotnet run transformer perf"
) else if "%choice%"=="3" (
    echo.
    echo Goodbye!
    exit /b 0
) else (
    echo.
    echo Starting Model Recommendations Demo...
    start cmd /k "dotnet run transformer models"
)

echo.
echo Demo launched in new window!
echo Press any key to exit...
pause >nul
