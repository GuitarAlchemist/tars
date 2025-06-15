@echo off
REM TARS CLI Launcher
REM Provides access to comprehensive TARS functionality including service management

set TARS_DIR=%~dp0
set TARS_EXE=%TARS_DIR%TarsEngine.FSharp.Cli\bin\Release\net9.0\TarsEngine.FSharp.Cli.exe

REM Check if the TARS CLI executable exists
if not exist "%TARS_EXE%" (
    echo ❌ TARS CLI not found!
    echo    Please build the TARS CLI first:
    echo    dotnet build TarsEngine.FSharp.Cli --configuration Release
    echo.
    echo    Alternative debug build:
    set TARS_EXE=%TARS_DIR%TarsEngine.FSharp.Cli\bin\Debug\net9.0\TarsEngine.FSharp.Cli.exe
    if not exist "%TARS_EXE%" (
        echo    dotnet build TarsEngine.FSharp.Cli --configuration Debug
        exit /b 1
    )
)

REM Pass all arguments to the TARS CLI executable
"%TARS_EXE%" %*
