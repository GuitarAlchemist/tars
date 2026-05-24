# Script to run the demo auto-improvement pipeline using Tree-of-Thought reasoning

Write-Host "Running demo auto-improvement pipeline using Tree-of-Thought reasoning..."

# Create the Samples directory if it doesn't exist
if (-not (Test-Path -Path "Samples")) {
    New-Item -Path "Samples" -ItemType Directory | Out-Null
}

# Run the demo auto-improvement pipeline command
Write-Host "Running demo auto-improvement pipeline command..."
dotnet run --project TarsCli demo-auto-improve --file Samples/SampleCode.cs --type performance --output demo_auto_improvement_report.md

Write-Host "Demo auto-improvement pipeline completed successfully!"
