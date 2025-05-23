#!/usr/bin/env pwsh
#Requires -Version 7.0

<#
.SYNOPSIS
    Demonstrates TARS capabilities by running various TARS CLI commands.

.DESCRIPTION
    This script showcases the key features of TARS (Transformative Autonomous Reasoning System)
    by running a series of TARS CLI commands that are known to work correctly.

.EXAMPLE
    .\Working-Demo.ps1

.NOTES
    Author: TARS Team
    Version: 1.0
    Date: 2025-03-31
#>

# Configuration
$TarsCliPath = Join-Path $PSScriptRoot "..\TarsCli\bin\Debug\net9.0\tarscli.exe"
$DemoDelay = 1.5  # Delay between commands in seconds
$PauseBetweenSections = 3  # Pause between major sections in seconds
$DemoTopic = "Artificial Intelligence"  # Default topic for demonstrations

# Save current directory and change to repository root
$CurrentDir = Get-Location
Set-Location (Join-Path $PSScriptRoot "..")

# ANSI color codes for pretty output
$Colors = @{
    Reset = "`e[0m"
    Bold = "`e[1m"
    Dim = "`e[2m"
    Underline = "`e[4m"
    Blink = "`e[5m"
    Reverse = "`e[7m"
    Hidden = "`e[8m"

    Black = "`e[30m"
    Red = "`e[31m"
    Green = "`e[32m"
    Yellow = "`e[33m"
    Blue = "`e[34m"
    Magenta = "`e[35m"
    Cyan = "`e[36m"
    White = "`e[37m"

    BgBlack = "`e[40m"
    BgRed = "`e[41m"
    BgGreen = "`e[42m"
    BgYellow = "`e[43m"
    BgBlue = "`e[44m"
    BgMagenta = "`e[45m"
    BgCyan = "`e[46m"
    BgWhite = "`e[47m"
}

# Check if TARS CLI exists
if (-not (Test-Path $TarsCliPath)) {
    Write-Host "$($Colors.Red)$($Colors.Bold)Error: TARS CLI not found at $TarsCliPath$($Colors.Reset)"
    Write-Host "Please build the TARS CLI project first by running 'dotnet build' in the TarsCli directory."
    exit 1
}

# Helper functions
function Write-Header {
    param (
        [string]$Text
    )

    $width = $Host.UI.RawUI.WindowSize.Width
    $padding = [math]::Max(0, [math]::Floor(($width - $Text.Length - 4) / 2))
    $line = "=" * $width

    Write-Host ""
    Write-Host "$($Colors.Cyan)$line$($Colors.Reset)"
    Write-Host "$($Colors.Cyan)$(" " * $padding)$($Colors.Bold)$($Colors.Yellow)$Text$($Colors.Reset)$($Colors.Cyan)$(" " * $padding)$($Colors.Reset)"
    Write-Host "$($Colors.Cyan)$line$($Colors.Reset)"
    Write-Host ""
}

function Write-SubHeader {
    param (
        [string]$Text
    )

    Write-Host ""
    Write-Host "$($Colors.Green)$($Colors.Bold)$Text$($Colors.Reset)"
    Write-Host "$($Colors.Green)$("-" * $Text.Length)$($Colors.Reset)"
    Write-Host ""
}

function Write-Command {
    param (
        [string]$Command
    )

    Write-Host "$($Colors.Yellow)> $($Colors.Bold)$Command$($Colors.Reset)"
    Start-Sleep -Seconds $DemoDelay
}

function Invoke-TarsCommand {
    param (
        [string]$Command,
        [switch]$NoHeader
    )

    if (-not $NoHeader) {
        Write-Command "tarscli $Command"
    }

    & $TarsCliPath $Command.Split(" ")
}

function Wait-ForNextSection {
    param (
        [int]$Seconds = $PauseBetweenSections,
        [string]$Message = "Continuing in"
    )

    for ($i = $Seconds; $i -gt 0; $i--) {
        Write-Host "`r$($Colors.Dim)$Message $i seconds...$($Colors.Reset)" -NoNewline
        Start-Sleep -Seconds 1
    }

    Write-Host "`r$(" " * 50)" -NoNewline
    Write-Host "`r" -NoNewline
}

# Main demo script
Clear-Host
Write-Header "TARS WORKING DEMONSTRATION"

Write-Host "$($Colors.White)This script will demonstrate the key features of TARS (Transformative Autonomous Reasoning System).$($Colors.Reset)"
Write-Host "$($Colors.White)Press Ctrl+C at any time to exit the demo.$($Colors.Reset)"
Write-Host ""
Write-Host "$($Colors.Yellow)Default demo topic: $($Colors.Bold)$DemoTopic$($Colors.Reset)"
Write-Host ""

$customTopic = Read-Host "Enter a custom topic for the demo (or press Enter to use the default)"
if (-not [string]::IsNullOrWhiteSpace($customTopic)) {
    $DemoTopic = $customTopic
    Write-Host "$($Colors.Green)Using custom topic: $($Colors.Bold)$DemoTopic$($Colors.Reset)"
}

Wait-ForNextSection -Message "Starting demo in"

# Section 1: Basic Information
Write-Header "BASIC INFORMATION"

Write-SubHeader "TARS Version"
Invoke-TarsCommand "--version"

Write-SubHeader "TARS Help"
Invoke-TarsCommand "--help"

Wait-ForNextSection

# Section 2: Chat Bot
Write-Header "CHAT BOT"

Write-SubHeader "Interactive Chat"
Invoke-TarsCommand "chat start"

Wait-ForNextSection

# Section 3: Console Capture
Write-Header "CONSOLE CAPTURE"

Write-SubHeader "Console Capture Demo"
Invoke-TarsCommand "console-capture --start"

Write-Host "$($Colors.Cyan)Generating some console output...$($Colors.Reset)"
Write-Host "Warning: This is a test warning message"
Write-Host "Error: This is a test error message" -ForegroundColor Red
Write-Host "Info: This is a test info message" -ForegroundColor Green
Write-Host "Debug: This is a test debug message" -ForegroundColor Yellow

Invoke-TarsCommand "console-capture --stop"

Wait-ForNextSection

# Section 4: Demo Mode
Write-Header "DEMO MODE"

Write-SubHeader "Running TARS Demo"
Invoke-TarsCommand "demo --type chatbot"

# Conclusion
Write-Header "DEMO COMPLETE"

Write-Host "$($Colors.Green)$($Colors.Bold)Thank you for exploring TARS capabilities!$($Colors.Reset)"
Write-Host ""
Write-Host "$($Colors.White)For more information, visit:$($Colors.Reset)"
Write-Host "$($Colors.Cyan)https://github.com/GuitarAlchemist/tars$($Colors.Reset)"
Write-Host ""
Write-Host "$($Colors.Yellow)To run specific commands, use:$($Colors.Reset)"
Write-Host "$($Colors.White)$TarsCliPath [command] [options]$($Colors.Reset)"
Write-Host ""

# Return to the original directory
Set-Location $CurrentDir
