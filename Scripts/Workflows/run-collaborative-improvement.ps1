# Run Collaborative Improvement between Augment Code and TARS CLI

# Step 1: Ensure TARS MCP service is running
Write-Host "Starting TARS MCP service..." -ForegroundColor Cyan
$mcpProcess = Start-Process -FilePath ".\tarscli.cmd" -ArgumentList "mcp start --url http://localhost:9000/" -PassThru -NoNewWindow

# Wait for the MCP service to initialize
Write-Host "Waiting for MCP service to initialize..." -ForegroundColor Cyan
Start-Sleep -Seconds 10

# Step 2: Run the collaborative improvement script
Write-Host "Starting collaborative improvement process..." -ForegroundColor Green
try {
    # Check if Node.js is installed
    $nodeVersion = node --version
    Write-Host "Node.js version: $nodeVersion" -ForegroundColor Cyan

    # Run the collaborative improvement script
    Write-Host "Running collaborative improvement script..." -ForegroundColor Cyan
    node augment-tars-collaborative-improvement.js

    Write-Host "Collaborative improvement process completed." -ForegroundColor Green
} catch {
    Write-Host "Error running collaborative improvement script: $_" -ForegroundColor Red
} finally {
    # Step 3: Stop the TARS MCP service when done
    Write-Host "Stopping TARS MCP service..." -ForegroundColor Cyan
    Start-Process -FilePath ".\tarscli.cmd" -ArgumentList "mcp stop --url http://localhost:9000/" -NoNewWindow -Wait

    Write-Host "Process completed." -ForegroundColor Green
}
