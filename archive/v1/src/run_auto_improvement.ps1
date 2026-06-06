# Script to run the auto-improvement pipeline using Metascript Tree-of-Thought reasoning

Write-Host "Running auto-improvement pipeline using Metascript Tree-of-Thought reasoning..."

# Create the Samples directory if it doesn't exist
if (-not (Test-Path -Path "Samples")) {
    New-Item -Path "Samples" -ItemType Directory | Out-Null
}

# Run the auto-improvement pipeline command
Write-Host "Running auto-improvement pipeline command..."
dotnet run --project TarsCli metascript-auto-improve --file Samples/SampleCode.cs --type performance --output auto_improvement_report.md

Write-Host "Auto-improvement pipeline completed successfully!"
