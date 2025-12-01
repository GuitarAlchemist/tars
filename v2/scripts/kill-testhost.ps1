$ErrorActionPreference = "SilentlyContinue"

Write-Host "Killing lingering testhost/dotnet from Tars.Tests..."
Get-Process testhost -ErrorAction SilentlyContinue | Stop-Process -Force
Get-Process dotnet -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -like "*\\tests\\Tars.Tests\\bin\\Debug\\net10.0\\testhost.exe" } |
    Stop-Process -Force

Write-Host "Done."
