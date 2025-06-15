# TARS CLI PowerShell Launcher
# Provides access to comprehensive TARS functionality including service management

param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Arguments
)

$TarsDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$TarsExe = Join-Path $TarsDir "TarsEngine.FSharp.Cli\bin\Release\net9.0\TarsEngine.FSharp.Cli.exe"

# Check if the TARS CLI executable exists
if (-not (Test-Path $TarsExe)) {
    Write-Host "❌ TARS CLI not found!" -ForegroundColor Red
    Write-Host "   Please build the TARS CLI first:" -ForegroundColor Yellow
    Write-Host "   dotnet build TarsEngine.FSharp.Cli --configuration Release" -ForegroundColor Yellow
    Write-Host ""

    # Try debug build as fallback
    $TarsExeDebug = Join-Path $TarsDir "TarsEngine.FSharp.Cli\bin\Debug\net9.0\TarsEngine.FSharp.Cli.exe"
    if (Test-Path $TarsExeDebug) {
        Write-Host "   Using debug build instead..." -ForegroundColor Yellow
        $TarsExe = $TarsExeDebug
    } else {
        Write-Host "   Alternative debug build:" -ForegroundColor Yellow
        Write-Host "   dotnet build TarsEngine.FSharp.Cli --configuration Debug" -ForegroundColor Yellow
        exit 1
    }
}

# Pass all arguments to the TARS CLI executable
& $TarsExe @Arguments
exit $LASTEXITCODE
