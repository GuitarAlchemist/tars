@echo off
setlocal

REM Find the TARS CLI executable
set TARS_CLI_PATH=TarsCli\bin\Debug\net9.0\tarscli.exe

if not exist %TARS_CLI_PATH% (
    echo TARS CLI not found at: %TARS_CLI_PATH%
    echo Please build the solution first.
    exit /b 1
)

echo Found TARS CLI at: %TARS_CLI_PATH%

REM Set environment variables to skip Ollama setup
set OLLAMA_SKIP_SETUP=true
set OLLAMA_USE_DOCKER=true

REM Check if Docker is running
docker ps >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Docker is not running. Please start Docker Desktop first.
    exit /b 1
)

REM Check if Ollama is running in Docker
docker ps | findstr "ollama" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Ollama is not running in Docker. Starting it...
    docker-compose -f docker-compose-simple.yml up -d
    timeout /t 5 /nobreak >nul
)

REM Run the TARS CLI with the provided arguments
%TARS_CLI_PATH% %*

endlocal
