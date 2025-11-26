$ErrorActionPreference = "Stop"

Write-Host "Building Tars..."
dotnet build

Write-Host "Running Demo Ping E2E..."
$projectPath = "src\Tars.Interface.Cli"
$output = dotnet run --project $projectPath --no-build -- demo-ping

$outputString = $output -join "`n"
Write-Host $outputString

if ($outputString -match "DemoAgent received: PING") {
    Write-Host "SUCCESS: Demo Ping verified." -ForegroundColor Green
} else {
    Write-Host "FAILURE: Demo Ping did not return expected output." -ForegroundColor Red
    exit 1
}
