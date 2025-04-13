#!/usr/bin/env pwsh
#Requires -Version 7.0

<#
.SYNOPSIS
    Run Autonomous Improvement of TARS Documentation and Codebase

.DESCRIPTION
    This script runs the autonomous improvement process for TARS, which includes:
    1. Registering the metascript with TARS
    2. Starting the TARS MCP service
    3. Running the collaborative improvement process
    4. Running the TARS metascript for autonomous improvement
    5. Stopping the TARS MCP service

.EXAMPLE
    .\run-autonomous-improvement.ps1

.NOTES
    Author: TARS Team
    Version: 1.0
#>

# Save current directory and change to repository root
$CurrentDir = Get-Location
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Step 1: Register the metascript with TARS
Write-Host "Registering metascript with TARS..." -ForegroundColor Cyan
& .\register-metascript.ps1

# Step 2: Start TARS MCP service
Write-Host "Starting TARS MCP service..." -ForegroundColor Cyan
$mcpProcess = Start-Process -FilePath ".\tarscli.cmd" -ArgumentList "mcp start --url http://localhost:8999/" -PassThru -NoNewWindow

# Wait for the MCP service to initialize
Write-Host "Waiting for MCP service to initialize..." -ForegroundColor Cyan
Start-Sleep -Seconds 10

# Step 3: Run the autonomous improvement process
Write-Host "Starting autonomous improvement process..." -ForegroundColor Green
try {
    # Check if the metascript exists
    $metascriptPath = "TarsCli/Metascripts/autonomous_improvement.tars"
    if (-not (Test-Path $metascriptPath)) {
        Write-Host "Error: Metascript not found at $metascriptPath" -ForegroundColor Red
        exit 1
    }

    # Run the TARS metascript
    Write-Host "Running TARS metascript for autonomous improvement..." -ForegroundColor Cyan
    & .\tarscli.cmd dsl run --file $metascriptPath --verbose

    Write-Host "TARS metascript execution completed." -ForegroundColor Green
} catch {
    Write-Host "Error in autonomous improvement process: $_" -ForegroundColor Red
} finally {
    # Step 4: Stop the TARS MCP service when done
    Write-Host "Stopping TARS MCP service..." -ForegroundColor Cyan
    Start-Process -FilePath ".\tarscli.cmd" -ArgumentList "mcp stop --url http://localhost:8999/" -NoNewWindow -Wait

    # Restore original directory
    Set-Location $CurrentDir

    Write-Host "Autonomous improvement process completed." -ForegroundColor Green
}
