@echo off
setlocal enabledelayedexpansion
REM TARS CLI Global Launcher
REM This script can be placed in PATH to run TARS from anywhere

REM Try to find TARS installation directory
set TARS_DIR=
if exist "%~dp0tars.cmd" (
    set TARS_DIR=%~dp0
) else if exist "C:\Users\spare\source\repos\tars\tars.cmd" (
    set TARS_DIR=C:\Users\spare\source\repos\tars\
) else (
    echo ❌ TARS installation not found!
    echo    Please ensure TARS is installed and tars.cmd exists
    exit /b 1
)

REM Call the main TARS script
call "!TARS_DIR!tars.cmd" %*
