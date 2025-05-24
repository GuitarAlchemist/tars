# Script to run the Simple Tree-of-Thought test

Write-Host "Running Simple Tree-of-Thought test..."

# Create the Samples directory if it doesn't exist
if (-not (Test-Path -Path "Samples")) {
    New-Item -Path "Samples" -ItemType Directory | Out-Null
}

# Create the output directory if it doesn't exist
if (-not (Test-Path -Path "tot_results")) {
    New-Item -Path "tot_results" -ItemType Directory | Out-Null
}

# Ensure the F# module directory exists
$fsharpDir = Join-Path -Path (Get-Location) -ChildPath "TarsEngine\FSharp"
if (-not (Test-Path -Path $fsharpDir)) {
    New-Item -Path $fsharpDir -ItemType Directory | Out-Null
}

# Run the Simple Tree-of-Thought command
Write-Host "Analyzing code..."
dotnet run --project TarsCli simple-tot --file Samples/SampleCode.cs --mode analyze

Write-Host "Generating fixes..."
dotnet run --project TarsCli simple-tot --file Samples/SampleCode.cs --mode generate

Write-Host "Applying fixes..."
dotnet run --project TarsCli simple-tot --file Samples/SampleCode.cs --mode apply

Write-Host "Simple Tree-of-Thought test completed successfully!"
