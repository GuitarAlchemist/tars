param (
    [Parameter(Mandatory=$true)]
    [string]$FilePath,
    
    [Parameter(Mandatory=$false)]
    [string]$OutputPath,
    
    [Parameter(Mandatory=$false)]
    [string]$Model = "llama3"
)

# Check if file exists
if (-not (Test-Path $FilePath)) {
    Write-Host "Error: File not found: $FilePath" -ForegroundColor Red
    exit 1
}

# Determine output path
if (-not $OutputPath) {
    $directory = [System.IO.Path]::GetDirectoryName($FilePath)
    $filename = [System.IO.Path]::GetFileNameWithoutExtension($FilePath)
    $extension = [System.IO.Path]::GetExtension($FilePath)
    $OutputPath = Join-Path $directory "$($filename)_improved$extension"
}

# Step 1: Run self-propose to generate improvement suggestions
Write-Host "Step 1: Generating improvement suggestions for $FilePath..." -ForegroundColor Cyan
dotnet run --project TarsCli/TarsCli.csproj -- self-propose --file $FilePath --model $Model

# Step 2: Run self-improve to apply the improvements
Write-Host "`nStep 2: Applying improvements to $FilePath..." -ForegroundColor Cyan
dotnet run --project TarsCli/TarsCli.csproj -- self-improve --file $FilePath --output $OutputPath --model $Model

# Step 3: Show the differences
if (Test-Path $OutputPath) {
    Write-Host "`nStep 3: Showing differences between original and improved files..." -ForegroundColor Cyan
    
    Write-Host "`nOriginal file (first 10 lines):" -ForegroundColor Yellow
    Get-Content $FilePath -TotalCount 10 | ForEach-Object { Write-Host "  $_" }
    
    Write-Host "`nImproved file (first 10 lines):" -ForegroundColor Green
    Get-Content $OutputPath -TotalCount 10 | ForEach-Object { Write-Host "  $_" }
    
    Write-Host "`nImproved file saved to: $OutputPath" -ForegroundColor Green
} else {
    Write-Host "`nError: Failed to generate improved file." -ForegroundColor Red
}

Write-Host "`nTARS Retroaction completed!" -ForegroundColor Cyan
