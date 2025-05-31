@echo off
echo ?? TARS Calculator Web App Launcher
echo ===================================
echo.

echo ?? Checking Node.js installation...
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ? Node.js not found! Please install Node.js from https://nodejs.org/
    echo.
    pause
    exit /b 1
)

echo ? Node.js found
echo.

echo ?? Installing dependencies...
call npm install --silent

if %errorlevel% neq 0 (
    echo ? Failed to install dependencies
    pause
    exit /b 1
)

echo ? Dependencies installed
echo.

echo ?? Starting calculator web app...
echo ?? Opening browser to: http://localhost:8080
echo ?? Press Ctrl+C to stop the server
echo.

start http://localhost:8080
call npx live-server --port=8080 --no-browser

pause
