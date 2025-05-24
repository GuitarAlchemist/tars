# Script to run the basic auto-improvement pipeline using Tree-of-Thought reasoning

Write-Host "Running basic auto-improvement pipeline using Tree-of-Thought reasoning..."

# Create the Samples directory if it doesn't exist
if (-not (Test-Path -Path "Samples")) {
    New-Item -Path "Samples" -ItemType Directory | Out-Null
}

# Run the basic auto-improvement pipeline command
Write-Host "Running basic auto-improvement pipeline command..."
dotnet run --project TarsCli basic-auto-improve --file Samples/SampleCode.cs --type performance --output basic_auto_improvement_report.md

Write-Host "Basic auto-improvement pipeline completed successfully!"
