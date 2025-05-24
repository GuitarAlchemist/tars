# Script to run the Metascript Tree-of-Thought pipeline

Write-Host "Running Metascript Tree-of-Thought pipeline..."

# Create the Samples directory if it doesn't exist
if (-not (Test-Path -Path "Samples")) {
    New-Item -Path "Samples" -ItemType Directory | Out-Null
}

# Create the output directory if it doesn't exist
if (-not (Test-Path -Path "tot_output")) {
    New-Item -Path "tot_output" -ItemType Directory | Out-Null
}

# Ensure the F# module directories exist
$fsharpDir = Join-Path -Path (Get-Location) -ChildPath "TarsEngine\FSharp"
if (-not (Test-Path -Path $fsharpDir)) {
    New-Item -Path $fsharpDir -ItemType Directory | Out-Null
}

# Register the services in Program.cs
Write-Host "Registering services..."
# This is just a placeholder - in a real implementation, you would modify Program.cs

# Run the Metascript Tree-of-Thought pipeline command
Write-Host "Running pipeline command..."
dotnet run --project TarsCli metascript-tot pipeline --template Samples/template.tars --values Samples/values.txt --output-dir tot_output

Write-Host "Metascript Tree-of-Thought pipeline completed successfully!"
