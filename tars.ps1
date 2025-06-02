# TARS Service Manager PowerShell Launcher
# Provides easy access to TARS Windows service management

param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Arguments
)

$TarsDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$TarsExe = Join-Path $TarsDir "TarsServiceManager\bin\Release\net9.0\tars.exe"

# Check if the TARS executable exists
if (-not (Test-Path $TarsExe)) {
    Write-Host "❌ TARS service manager not found!" -ForegroundColor Red
    Write-Host "   Please build the service manager first:" -ForegroundColor Yellow
    Write-Host "   dotnet build TarsServiceManager --configuration Release" -ForegroundColor Yellow
    exit 1
}

# Pass all arguments to the TARS executable
& $TarsExe @Arguments
exit $LASTEXITCODE
