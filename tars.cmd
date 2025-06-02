@echo off
REM TARS Service Manager Launcher
REM Provides easy access to TARS Windows service management

set TARS_DIR=%~dp0
set TARS_EXE=%TARS_DIR%TarsServiceManager\bin\Release\net9.0\tars.exe

REM Check if the TARS executable exists
if not exist "%TARS_EXE%" (
    echo ❌ TARS service manager not found!
    echo    Please build the service manager first:
    echo    dotnet build TarsServiceManager --configuration Release
    exit /b 1
)

REM Pass all arguments to the TARS executable
"%TARS_EXE%" %*
