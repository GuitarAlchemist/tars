# Run Autonomous Improvement of TARS Documentation and Codebase

# Step 1: Register the metascript with TARS
Write-Host "Registering metascript with TARS..." -ForegroundColor Cyan
& .\register-metascript.ps1

# Step 2: Start TARS MCP service
Write-Host "Starting TARS MCP service..." -ForegroundColor Cyan
$mcpProcess = Start-Process -FilePath ".\tarscli.cmd" -ArgumentList "mcp start --url http://localhost:9000/" -PassThru -NoNewWindow

# Wait for the MCP service to initialize
Write-Host "Waiting for MCP service to initialize..." -ForegroundColor Cyan
Start-Sleep -Seconds 10

# Step 3: Run the collaborative improvement script
Write-Host "Starting collaborative improvement process..." -ForegroundColor Green
try {
    # Check if Node.js is installed
    $nodeVersion = node --version
    Write-Host "Node.js version: $nodeVersion" -ForegroundColor Cyan

    # Run the collaborative improvement script
    Write-Host "Running collaborative improvement script..." -ForegroundColor Cyan
    node augment-tars-collaborative-improvement.js

    Write-Host "Collaborative improvement process completed." -ForegroundColor Green

    # Step 4: Run the TARS metascript
    Write-Host "Running TARS metascript for autonomous improvement..." -ForegroundColor Cyan
    & .\tarscli.cmd dsl run --file TarsCli/Metascripts/autonomous_improvement.tars --verbose

    Write-Host "TARS metascript execution completed." -ForegroundColor Green
} catch {
    Write-Host "Error in autonomous improvement process: $_" -ForegroundColor Red
} finally {
    # Step 5: Stop the TARS MCP service when done
    Write-Host "Stopping TARS MCP service..." -ForegroundColor Cyan
    Start-Process -FilePath ".\tarscli.cmd" -ArgumentList "mcp stop --url http://localhost:9000/" -NoNewWindow -Wait

    Write-Host "Autonomous improvement process completed." -ForegroundColor Green
}
