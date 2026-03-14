@echo off
echo Checking if TARS MCP Server is running...

REM Try to connect to the MCP server
powershell -Command "try { $response = Invoke-WebRequest -Uri 'http://localhost:9000/status' -Method HEAD -TimeoutSec 2; exit 0 } catch { exit 1 }"

REM Check the exit code
if %ERRORLEVEL% EQU 0 (
    echo TARS MCP Server is already running.
) else (
    echo TARS MCP Server is not running. Starting it...
    call start-tars-mcp.cmd
    echo Waiting for TARS MCP Server to start...
    timeout /t 5 /nobreak > nul
    echo TARS MCP Server should be running now.
)
