$url = "http://localhost:9000/"
$body = @{
    action = "knowledge"
    operation = "cycle"
    explorationDirectory = "docs/Explorations/v1/Chats"
    targetDirectory = "TarsCli/Services"
    pattern = "*.cs"
    model = "llama3"
} | ConvertTo-Json

Write-Host "Running knowledge improvement cycle..."
$response = Invoke-RestMethod -Uri $url -Method Post -Body $body -ContentType "application/json"

Write-Host "Response received:"
$response | ConvertTo-Json -Depth 10
