# TARS CLI Global PowerShell Launcher
# This script can be placed in PATH to run TARS from anywhere

param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Arguments
)

# Try to find TARS installation directory
$TarsDir = $null
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

if (Test-Path (Join-Path $ScriptDir "tars.ps1")) {
    $TarsDir = $ScriptDir
} elseif (Test-Path "C:\Users\spare\source\repos\tars\tars.ps1") {
    $TarsDir = "C:\Users\spare\source\repos\tars"
} else {
    Write-Host "❌ TARS installation not found!" -ForegroundColor Red
    Write-Host "   Please ensure TARS is installed and tars.ps1 exists" -ForegroundColor Yellow
    exit 1
}

# Call the main TARS script
$TarsScript = Join-Path $TarsDir "tars.ps1"
& $TarsScript @Arguments
exit $LASTEXITCODE
