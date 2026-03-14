@echo off
setlocal enabledelayedexpansion

REM TARS Auto-Coding Demo
REM This script demonstrates TARS Auto-Coding capabilities

REM Display the TARS logo
echo.
echo   ████████╗ █████╗ ██████╗ ███████╗
echo   ╚══██╔══╝██╔══██╗██╔══██╗██╔════╝
echo      ██║   ███████║██████╔╝███████╗
echo      ██║   ██╔══██║██╔══██╗╚════██║
echo      ██║   ██║  ██║██║  ██║███████║
echo      ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝
echo.
echo   Transformative Autonomous Reasoning System
echo   Auto-Coding Demo
echo.
echo   This demo showcases the auto-coding capabilities of TARS.
echo   Press Ctrl+C at any time to exit the demo.
echo.
echo   Press Enter to begin...
pause > nul

REM Check if Docker is running
echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo   Checking Prerequisites
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.

docker ps > nul 2>&1
if %errorlevel% neq 0 (
    echo   Docker is not running. Please start Docker Desktop first.
    exit /b 1
) else (
    echo   Docker is running
)

REM Check if the Docker network exists
docker network ls | findstr "tars-network" > nul
if %errorlevel% neq 0 (
    echo   Creating Docker network 'tars-network'...
    docker network create tars-network
    echo   Docker network 'tars-network' created
) else (
    echo   Docker network 'tars-network' exists
)

REM Section 1: Docker Auto-Coding
echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo   Docker Auto-Coding
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.

echo.
echo   Simple Auto-Coding
echo   ─────────────────
echo.

echo   Running a simple auto-coding demo
echo   ^> .\Scripts\AutoCoding\Demos\Test-SwarmAutoCode-Simple.ps1
echo.
echo   Press Enter to continue...
pause > nul

echo.
powershell -ExecutionPolicy Bypass -File ".\Scripts\AutoCoding\Demos\Test-SwarmAutoCode-Simple.ps1"
echo.

REM Section 2: Swarm Auto-Coding
echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo   Swarm Auto-Coding
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.

echo.
echo   Swarm Auto-Coding Test
echo   ─────────────────────
echo.

echo   Running a swarm auto-coding test
echo   ^> .\Scripts\AutoCoding\Demos\Run-SwarmAutoCode-Test.ps1
echo.
echo   Press Enter to continue...
pause > nul

echo.
powershell -ExecutionPolicy Bypass -File ".\Scripts\AutoCoding\Demos\Run-SwarmAutoCode-Test.ps1"
echo.

REM Section 3: Auto-Coding with TARS CLI
echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo   Auto-Coding with TARS CLI
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.

echo.
echo   Docker Auto-Coding
echo   ─────────────────
echo.

echo   Running Docker auto-coding with TARS CLI
echo   ^> dotnet run --project TarsCli/TarsCli.csproj -- auto-code --docker --demo
echo.
echo   Press Enter to continue...
pause > nul

echo.
dotnet run --project TarsCli/TarsCli.csproj -- auto-code --docker --demo
echo.

echo.
echo   Swarm Auto-Coding
echo   ────────────────
echo.

echo   Running Swarm auto-coding with TARS CLI
echo   ^> dotnet run --project TarsCli/TarsCli.csproj -- auto-code --swarm --demo
echo.
echo   Press Enter to continue...
pause > nul

echo.
dotnet run --project TarsCli/TarsCli.csproj -- auto-code --swarm --demo
echo.

REM Conclusion
echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo   Conclusion
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.

echo   TARS Auto-Coding Demo Completed
echo.
echo   You've seen how TARS can auto-code itself using Docker containers and swarm architecture.
echo   For more information, see the documentation in the docs/AutoCoding directory.
echo.
echo   Thank you for exploring TARS Auto-Coding capabilities!
echo.

endlocal
