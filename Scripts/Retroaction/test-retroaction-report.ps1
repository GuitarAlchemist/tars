$url = "http://localhost:9000/"
$body = @{
    action = "knowledge"
    operation = "retroaction"
    explorationDirectory = "docs/Explorations/v1/Chats"
    targetDirectory = "TarsCli/Services"
    model = "llama3"
} | ConvertTo-Json

Write-Host "Generating retroaction report..."
$response = Invoke-RestMethod -Uri $url -Method Post -Body $body -ContentType "application/json"

Write-Host "Response received:"
$response | ConvertTo-Json -Depth 10
