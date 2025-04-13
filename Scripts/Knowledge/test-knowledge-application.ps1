$url = "http://localhost:9000/"
$body = @{
    action = "knowledge"
    operation = "apply"
    filePath = "TarsCli/Services/DslService.cs"
    model = "llama3"
} | ConvertTo-Json

Write-Host "Applying knowledge to file..."
$response = Invoke-RestMethod -Uri $url -Method Post -Body $body -ContentType "application/json"

Write-Host "Response received:"
$response | ConvertTo-Json -Depth 10
