@echo off
echo ?? TARS Project Launcher: enhanced_demo_214018
echo ================================
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

echo ?? Starting enhanced demo project...
echo ?? Project type: web
echo.

if not exist package.json (
    echo Creating package.json...
    echo {"name": "enhanced_demo_214018", "version": "1.0.0", "scripts": {"start": "npx live-server src --port=3000"}} > package.json
)

call npm install --silent
echo ?? Opening browser to: http://localhost:3000
start http://localhost:3000
call npx live-server src --port=3000
