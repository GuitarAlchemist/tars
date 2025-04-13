$url = "http://localhost:9000/"
$body = @{
    action = "knowledge"
    operation = "extract"
    filePath = "docs/Explorations/v1/Chats/ChatGPT-TARS Project Implications.md"
    model = "llama3"
} | ConvertTo-Json

Write-Host "Extracting knowledge from file..."
$response = Invoke-RestMethod -Uri $url -Method Post -Body $body -ContentType "application/json"

Write-Host "Response received:"
$response | ConvertTo-Json -Depth 10
