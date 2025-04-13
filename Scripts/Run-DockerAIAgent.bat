@echo off
REM Run Docker AI Agent with TARS integration
REM This batch file runs the PowerShell script with the same name

REM Set console code page to UTF-8 to handle special characters correctly
chcp 65001 > nul

REM Run the PowerShell script
powershell -ExecutionPolicy Bypass -File "%~dp0Run-DockerAIAgent.ps1"

REM Pause at the end to see the output
pause
