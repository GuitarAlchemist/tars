# Simple script to fix IDE compilation issues
# This cleans IDE caches and rebuilds the solution

Write-Host "Fixing IDE compilation issues..." -ForegroundColor Cyan

# Step 1: Clean Visual Studio cache
Write-Host ""
Write-Host "Step 1: Cleaning Visual Studio cache..." -ForegroundColor Yellow
if (Test-Path ".vs") {
    Remove-Item -Path ".vs" -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "  Removed .vs folder" -ForegroundColor Green
} else {
    Write-Host "  No .vs folder found" -ForegroundColor Gray
}

# Step 2: Clean all bin and obj folders
Write-Host ""
Write-Host "Step 2: Cleaning build artifacts..." -ForegroundColor Yellow
$cleaned = 0
Get-ChildItem -Path . -Include bin,obj -Recurse -Directory | ForEach-Object {
    Remove-Item -Path $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
    $cleaned++
}
Write-Host "  Cleaned $cleaned build folders" -ForegroundColor Green

# Step 3: Restore packages
Write-Host ""
Write-Host "Step 3: Restoring NuGet packages..." -ForegroundColor Yellow
dotnet restore Tars.sln
if ($LASTEXITCODE -eq 0) {
    Write-Host "  Packages restored successfully" -ForegroundColor Green
} else {
    Write-Host "  Package restore had issues but continuing..." -ForegroundColor Yellow
}

# Step 4: Build the solution
Write-Host ""
Write-Host "Step 4: Building solution..." -ForegroundColor Yellow
dotnet build Tars.sln -c Debug
if ($LASTEXITCODE -eq 0) {
    Write-Host "  Build successful" -ForegroundColor Green
} else {
    Write-Host "  Build had errors - see above" -ForegroundColor Red
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "IDE fix complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Close Visual Studio completely"
Write-Host "2. Reopen Visual Studio"
Write-Host "3. Open Tars.sln"
Write-Host "4. Build -> Rebuild Solution"
Write-Host ""
Write-Host "The solution should now compile in the IDE." -ForegroundColor Green

