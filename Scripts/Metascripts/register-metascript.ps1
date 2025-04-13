# Register the autonomous improvement metascript with TARS

# Step 1: Copy the metascript to the TARS metascripts directory
$metascriptSource = "autonomous_improvement.tars"
$metascriptDestination = "C:\Users\spare\source\repos\tars\TarsCli\Metascripts\autonomous_improvement.tars"

Write-Host "Copying metascript to TARS metascripts directory..." -ForegroundColor Cyan
Copy-Item -Path $metascriptSource -Destination $metascriptDestination -Force

Write-Host "Metascript registered successfully." -ForegroundColor Green
