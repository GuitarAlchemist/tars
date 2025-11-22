@echo off
echo.
echo ========================================
echo   TARS Mixture of Experts Demo
echo ========================================
echo.
echo This demonstrates something EXTRAORDINARY:
echo A real MoE system using downloaded transformer models!
echo.
echo Choose a demo:
echo   1. Expert Status (see available models)
echo   2. MoE Architecture (system design)
echo   3. Task Execution (intelligent routing)
echo   4. Download More Models
echo   5. Exit
echo.
set /p choice="Enter your choice (1-5): "

if "%choice%"=="2" (
    echo.
    echo Starting MoE Architecture Demo...
    start cmd /k "dotnet run moe architecture"
) else if "%choice%"=="3" (
    echo.
    echo Starting Task Execution Demo...
    start cmd /k "dotnet run moe execute"
) else if "%choice%"=="4" (
    echo.
    echo Starting Model Download...
    start cmd /k "dotnet run transformer download"
) else if "%choice%"=="5" (
    echo.
    echo Goodbye!
    exit /b 0
) else (
    echo.
    echo Starting Expert Status Demo...
    start cmd /k "dotnet run moe status"
)

echo.
echo Demo launched in new window!
echo Press any key to exit...
pause >nul
