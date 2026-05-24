# Script to reload the TARS solution in Visual Studio
# This helps refresh the IDE's understanding of the project structure

Write-Host "Reloading TARS solution..." -ForegroundColor Cyan

# Clean bin and obj folders
Write-Host "Cleaning build artifacts..." -ForegroundColor Yellow
Get-ChildItem -Path . -Include bin,obj -Recurse -Directory | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# Restore packages
Write-Host "Restoring NuGet packages..." -ForegroundColor Yellow
dotnet restore Tars.sln

Write-Host "`nSolution reloaded. Please close and reopen Visual Studio." -ForegroundColor Green
Write-Host "Or use: File -> Close Solution, then File -> Open -> Project/Solution" -ForegroundColor Cyan

