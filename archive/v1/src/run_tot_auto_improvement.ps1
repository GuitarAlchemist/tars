# Script to run the Tree-of-Thought auto-improvement pipeline

Write-Host "Running Tree-of-Thought auto-improvement pipeline..."

# Create the Samples directory if it doesn't exist
if (-not (Test-Path -Path "Samples")) {
    New-Item -Path "Samples" -ItemType Directory | Out-Null
}

# Create the output directory if it doesn't exist
if (-not (Test-Path -Path "tot_output")) {
    New-Item -Path "tot_output" -ItemType Directory | Out-Null
}

# Create the Metascripts directory if it doesn't exist
if (-not (Test-Path -Path "Metascripts\TreeOfThought")) {
    New-Item -Path "Metascripts\TreeOfThought" -ItemType Directory -Force | Out-Null
}

# Run the Tree-of-Thought auto-improvement pipeline command
Write-Host "Running Tree-of-Thought auto-improvement pipeline command..."
dotnet run --project TarsCli tot-auto-improve --file Samples/SampleCode.cs --type performance --output-dir tot_output

Write-Host "Tree-of-Thought auto-improvement pipeline completed successfully!"
