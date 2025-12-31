$proc = Start-Process -FilePath "dotnet" -ArgumentList "run --project src/Tars.Interface.Cli/Tars.Interface.Cli.fsproj -- chat" -PassThru -NoNewWindow
Write-Host "Starting TARS chat..."
Start-Sleep -Seconds 10

try {
    Write-Host "Checking health..."
    $resp = Invoke-WebRequest "http://localhost:9090/health"
    Write-Host "Health Response: $($resp.StatusCode)"
    Write-Host $resp.Content
    
    Write-Host "Checking metrics..."
    $metrics = Invoke-WebRequest "http://localhost:9090/metrics"
    Write-Host "Metrics Response: $($metrics.StatusCode)"
    Write-Host ($metrics.Content | Select-Object -First 5)
}
catch {
    Write-Host "Error: $_"
}
finally {
    Write-Host "Stopping TARS..."
    Stop-Process -Id $proc.Id -Force
}
