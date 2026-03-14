@echo off
echo ?? TARS Enhanced Project: tars_enhanced_demo_20250529_212812
echo ========================================
echo.
echo ?? Checking Node.js...
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ? Node.js not found! Install from https://nodejs.org/
    pause
    exit /b 1
)
echo ? Node.js found
echo.
echo ?? Starting web application...
start http://localhost:3000
call npx live-server src --port=3000
