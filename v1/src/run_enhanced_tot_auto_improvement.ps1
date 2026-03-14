# Script to run the enhanced Tree-of-Thought auto-improvement pipeline

Write-Host "Running enhanced Tree-of-Thought auto-improvement pipeline..."

# Create the Samples directory if it doesn't exist
if (-not (Test-Path -Path "Samples")) {
    New-Item -Path "Samples" -ItemType Directory | Out-Null
}

# Create the output directory if it doesn't exist
if (-not (Test-Path -Path "enhanced_tot_output")) {
    New-Item -Path "enhanced_tot_output" -ItemType Directory | Out-Null
}

# Create the Metascripts directory if it doesn't exist
if (-not (Test-Path -Path "Metascripts\TreeOfThought")) {
    New-Item -Path "Metascripts\TreeOfThought" -ItemType Directory -Force | Out-Null
}

# Run the enhanced Tree-of-Thought auto-improvement pipeline command
Write-Host "Running enhanced Tree-of-Thought auto-improvement pipeline command..."
dotnet run --project TarsCli enhanced-tot --file Samples/SampleCode.cs --type performance --output enhanced_tot_output/enhanced_tot_report.md

Write-Host "Enhanced Tree-of-Thought auto-improvement pipeline completed successfully!"
