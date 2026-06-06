# Script to fix IDE compilation issues
# This cleans IDE caches and rebuilds the solution

Write-Host "Fixing IDE compilation issues..." -ForegroundColor Cyan

# Step 1: Clean Visual Studio cache
Write-Host ""
Write-Host "Step 1: Cleaning Visual Studio cache..." -ForegroundColor Yellow
if (Test-Path ".vs") {
    Remove-Item -Path ".vs" -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "  Removed .vs folder" -ForegroundColor Green
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

# Step 3: Clean NuGet cache for this solution
Write-Host ""
Write-Host "Step 3: Cleaning NuGet package cache..." -ForegroundColor Yellow
dotnet nuget locals all --clear
Write-Host "  NuGet cache cleared" -ForegroundColor Green

# Step 4: Restore packages
Write-Host ""
Write-Host "Step 4: Restoring NuGet packages..." -ForegroundColor Yellow
dotnet restore Tars.sln --force
if ($LASTEXITCODE -eq 0) {
    Write-Host "  Packages restored successfully" -ForegroundColor Green
} else {
    Write-Host "  Package restore failed" -ForegroundColor Red
    exit 1
}

# Step 5: Build the solution
Write-Host ""
Write-Host "Step 5: Building solution..." -ForegroundColor Yellow
dotnet build Tars.sln -c Debug
if ($LASTEXITCODE -eq 0) {
    Write-Host "  Build successful" -ForegroundColor Green
} else {
    Write-Host "  Build failed - see errors above" -ForegroundColor Red
    Write-Host ""
    Write-Host "Try opening Visual Studio now. The IDE should pick up the changes." -ForegroundColor Cyan
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "IDE fix complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Close Visual Studio if it is open"
Write-Host "2. Reopen Visual Studio"
Write-Host "3. Open Tars.sln"
Write-Host "4. Build -> Rebuild Solution"
Write-Host ""
Write-Host "The solution should now compile in the IDE." -ForegroundColor Green

