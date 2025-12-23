@echo off
echo.
echo ╔════════════════════════════════════════╗
echo ║       Starting TARS UI                 ║
echo ╚════════════════════════════════════════╝
echo.
echo URL: http://localhost:5000
echo.

cd /d "%~dp0"
dotnet run --project src/Tars.Interface.Ui/Tars.Interface.Ui.fsproj

pause
