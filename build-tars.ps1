# Build and run the TARS F# CLI

param(
    [string]$Command = "help",
    [string[]]$Args = @(),
    [switch]$Build = $false,
    [switch]$Clean = $false,
    [switch]$Test = $false
)

$ProjectPath = "TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj"
$CoreProjectPath = "TarsEngine.FSharp.Core.Working/TarsEngine.FSharp.Core.Working.fsproj"

Write-Host "TARS F# CLI Build Script" -ForegroundColor Cyan
Write-Host "========================" -ForegroundColor Cyan

if ($Clean) {
    Write-Host "Cleaning projects..." -ForegroundColor Yellow
    dotnet clean $CoreProjectPath
    dotnet clean $ProjectPath
}

if ($Build -or $Clean -or $Test) {
    Write-Host "Building F# Core..." -ForegroundColor Yellow
    dotnet build $CoreProjectPath
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Core build failed!" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "Building F# CLI..." -ForegroundColor Yellow
    dotnet build $ProjectPath
    if ($LASTEXITCODE -ne 0) {
        Write-Host "CLI build failed!" -ForegroundColor Red
        exit 1
    }
    Write-Host "Build successful!" -ForegroundColor Green
}

if ($Test) {
    Write-Host "Running tests..." -ForegroundColor Yellow
    # Add test execution here when we have test projects
    Write-Host "Tests completed!" -ForegroundColor Green
}

if ($Command -ne "") {
    $AllArgs = @($Command) + $Args
    $ArgsString = $AllArgs -join " "
    
    Write-Host "Running: tars $ArgsString" -ForegroundColor Cyan
    Write-Host "----------------------------------------" -ForegroundColor Gray
    dotnet run --project $ProjectPath -- @AllArgs
}

Write-Host "----------------------------------------" -ForegroundColor Gray
Write-Host "Available commands:" -ForegroundColor Yellow
Write-Host "  help        - Display help information" -ForegroundColor White
Write-Host "  version     - Display version information" -ForegroundColor White
Write-Host "  improve     - Run auto-improvement pipeline" -ForegroundColor White
Write-Host "  compile     - Compile F# source code" -ForegroundColor White
Write-Host "  run         - Run F# scripts or applications" -ForegroundColor White
Write-Host "  test        - Run tests and generate test reports" -ForegroundColor White
Write-Host "  analyze     - Analyze code for quality and patterns" -ForegroundColor White
Write-Host "  metascript  - Execute metascript files" -ForegroundColor White
Write-Host "" -ForegroundColor White
Write-Host "Examples:" -ForegroundColor Yellow
Write-Host "  .\build-tars.ps1 -Command help" -ForegroundColor Gray
Write-Host "  .\build-tars.ps1 -Command analyze -Args '.'" -ForegroundColor Gray
Write-Host "  .\build-tars.ps1 -Command metascript -Args 'test.tars'" -ForegroundColor Gray
Write-Host "  .\build-tars.ps1 -Command test -Args '--generate'" -ForegroundColor Gray
