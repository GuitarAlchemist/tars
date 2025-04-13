# Start-McpServer.ps1
# This script starts the MCP server

# Function to display colored text
function Write-ColorText {
    param (
        [string]$Text,
        [string]$Color = "White"
    )
    
    Write-Host $Text -ForegroundColor $Color
}

# Main script
Write-ColorText "Starting MCP Server" "Cyan"
Write-ColorText "=================" "Cyan"

# Start the MCP server
Write-ColorText "Starting MCP server..." "Yellow"
$tarsCli = "TarsCli\bin\Debug\net9.0\tarscli.exe"

if (Test-Path $tarsCli) {
    Write-ColorText "Found TARS CLI at: $tarsCli" "Green"
    
    # Start the MCP server
    Start-Process -FilePath $tarsCli -ArgumentList "mcp", "--server", "--auto-execute" -NoNewWindow
    
    Write-ColorText "MCP server started" "Green"
    Write-ColorText "MCP server is running at http://localhost:8999/" "Green"
}
else {
    Write-ColorText "TARS CLI not found at: $tarsCli" "Red"
    Write-ColorText "Please build the solution first" "Red"
    exit 1
}

Write-ColorText "MCP server setup completed" "Cyan"
