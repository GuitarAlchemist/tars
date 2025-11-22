@echo off
echo.
echo ========================================
echo   TARS CLI Swarm Demo
echo ========================================
echo.

cd /d "%~dp0"

echo Building TARS CLI...
dotnet build TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj --verbosity quiet

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ Build failed! Please check the errors above.
    pause
    exit /b 1
)

echo.
echo ✅ Build successful! 
echo.
echo 🚀 Demonstrating TARS CLI Swarm Features...
echo.

echo ========================================
echo 1. Full Demo (Header + Status + Tests)
echo ========================================
dotnet run --project TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj -- swarm demo

echo.
echo ========================================
echo 2. Container Status Only
echo ========================================
dotnet run --project TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj -- swarm status

echo.
echo ========================================
echo 3. Health Tests Only
echo ========================================
dotnet run --project TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj -- swarm test

echo.
echo ========================================
echo 4. Performance Monitor
echo ========================================
dotnet run --project TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj -- swarm monitor

echo.
echo ========================================
echo 5. Command Execution Demo
echo ========================================
dotnet run --project TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj -- swarm commands

echo.
echo ========================================
echo   Demo Complete!
echo ========================================
echo.
echo Available TARS CLI swarm commands:
echo   tars swarm demo      # Full demo with header
echo   tars swarm status    # Container status table
echo   tars swarm test      # Health test results
echo   tars swarm monitor   # Performance monitoring
echo   tars swarm commands  # Command execution demo
echo.
echo Press any key to exit...
pause > nul
