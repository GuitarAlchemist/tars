
$ErrorActionPreference = "Stop"

# Load secrets
if (-not (Test-Path "secrets.json")) {
    Write-Error "secrets.json not found!"
}

$secrets = Get-Content "secrets.json" | ConvertFrom-Json
$email = $secrets.OPENWEBUI_EMAIL
$password = $secrets.OPENWEBUI_PASSWORD
$baseUrl = "https://aialpha.bar-scouts.com/"

Write-Host "Authenticating to $baseUrl..."
$authBody = @{
    email    = $email
    password = $password
} | ConvertTo-Json

try {
    $authResponse = Invoke-RestMethod -Uri "$($baseUrl)api/v1/auths/signin" -Method Post -Body $authBody -ContentType "application/json" -SkipCertificateCheck
    $token = $authResponse.token
    Write-Host "Authentication successful! Token obtained."
}
catch {
    Write-Error "Authentication failed: $_"
}

# List Models
Write-Host "Listing models..."
try {
    $headers = @{
        Authorization = "Bearer $token"
    }
    # Try /ollama/api/tags
    $tagsUrl = "$($baseUrl)ollama/api/tags"
    $tagsResponse = Invoke-RestMethod -Uri $tagsUrl -Method Get -Headers $headers -SkipCertificateCheck
    
    Write-Host "Available Models:"
    $tagsResponse.models | ForEach-Object { Write-Host " - $($_.name)" }
    
    # Check if qwen2.5-coder:latest exists
    $targetModel = "qwen2.5-coder:latest"
    $found = $tagsResponse.models | Where-Object { $_.name -eq $targetModel }
    
    if ($found) {
        Write-Host "Target model '$targetModel' found!"
    }
    else {
        Write-Warning "Target model '$targetModel' NOT found. You may need to pull it or use a different model."
    }
}
catch {
    Write-Error "Failed to list models: $_"
}
