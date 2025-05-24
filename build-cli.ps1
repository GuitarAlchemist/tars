# Build and run the TARS F# CLI

param(
    [string]$Command = "help",
    [switch]$Build = $false,
    [switch]$Clean = $false
)

$ProjectPath = "TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj"

if ($Clean) {
    Write-Host "Cleaning project..." -ForegroundColor Yellow
    dotnet clean $ProjectPath
}

if ($Build -or $Clean) {
    Write-Host "Building project..." -ForegroundColor Yellow
    dotnet build $ProjectPath
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Build failed!" -ForegroundColor Red
        exit 1
    }
    Write-Host "Build successful!" -ForegroundColor Green
}

if ($Command -ne "") {
    Write-Host "Running: tars $Command" -ForegroundColor Cyan
    Write-Host "----------------------------------------" -ForegroundColor Gray
    dotnet run --project $ProjectPath -- $Command
}

Write-Host "----------------------------------------" -ForegroundColor Gray
Write-Host "Available commands:" -ForegroundColor Yellow
Write-Host "  help     - Display help information" -ForegroundColor White
Write-Host "  version  - Display version information" -ForegroundColor White
Write-Host "  improve  - Run auto-improvement pipeline" -ForegroundColor White
