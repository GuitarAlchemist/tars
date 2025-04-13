$url = "http://localhost:9000/"
$body = @{
    action = "tars"
    operation = "capabilities"
} | ConvertTo-Json

Write-Host "Sending request to TARS MCP server at $url..."
$response = Invoke-RestMethod -Uri $url -Method Post -Body $body -ContentType "application/json"

Write-Host "Response received:"
$response | ConvertTo-Json -Depth 10
