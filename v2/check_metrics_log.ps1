$proc = Start-Process -FilePath "dotnet" -ArgumentList "run --project src/Tars.Interface.Cli/Tars.Interface.Cli.fsproj -- chat" -RedirectStandardOutput "tars_log.txt" -RedirectStandardError "tars_err.txt" -PassThru -NoNewWindow
Write-Host "Starting TARS chat (logging to tars_log.txt)..."
Start-Sleep -Seconds 10

try {
    Write-Host "Checking health..."
    $resp = Invoke-WebRequest "http://localhost:9090/health"
    Write-Host "Health Response: $($resp.StatusCode)"
    
    Write-Host "Checking metrics..."
    $metrics = Invoke-WebRequest "http://localhost:9090/metrics"
    Write-Host "Metrics Response: $($metrics.StatusCode)"
}
catch {
    Write-Host "Error: $_"
}
finally {
    Write-Host "Stopping TARS..."
    Stop-Process -Id $proc.Id -Force
    
    Write-Host "--- LOG ---"
    Get-Content "tars_log.txt" -Tail 10
    Get-Content "tars_err.txt"
}
