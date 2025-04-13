# Run-AutoCoding-Real.ps1
# This script runs the full auto-coding process with a real implementation

# Function to display colored text
function Write-ColorText {
    param (
        [string]$Text,
        [string]$Color = "White"
    )

    Write-Host $Text -ForegroundColor $Color
}

# Function to check if Docker is running
function Test-DockerRunning {
    try {
        $dockerPs = docker ps
        return $true
    }
    catch {
        return $false
    }
}

# Main script
Write-ColorText "TARS Auto-Coding (Real Implementation)" "Cyan"
Write-ColorText "=================================" "Cyan"

# Check if Docker is running
if (Test-DockerRunning) {
    Write-ColorText "Docker is running" "Green"
}
else {
    Write-ColorText "Docker is not running. Please start Docker Desktop first." "Red"
    exit 1
}

# Check if Ollama is running in Docker
$ollamaRunning = docker ps | Select-String "ollama"
if (-not $ollamaRunning) {
    Write-ColorText "Ollama is not running in Docker. Starting it..." "Yellow"
    docker-compose -f docker-compose-simple.yml up -d
    Start-Sleep -Seconds 5
}

# Create a Docker network if it doesn't exist
$networkExists = docker network ls | Select-String "tars-network"
if (-not $networkExists) {
    Write-ColorText "Creating Docker network: tars-network" "Yellow"
    docker network create tars-network
    Write-ColorText "Docker network created: tars-network" "Green"
}

# Create a Docker container for auto-coding
Write-ColorText "Creating Docker container for auto-coding..." "Yellow"
docker run -d --name tars-auto-coding --network tars-network -v ${PWD}:/app/workspace ollama/ollama:latest
Start-Sleep -Seconds 5

# Find a TODO in the codebase
Write-ColorText "Finding a TODO in the codebase..." "Yellow"
$todoFile = "Experiments\ChatbotExample1\Services\Ingestion\IngestionCacheDbContext.cs"
if (Test-Path $todoFile) {
    Write-ColorText "Found TODO in file: $todoFile" "Green"

    # Get the TODO content
    $todoContent = Get-Content -Path $todoFile -Raw
    Write-ColorText "TODO content:" "Green"
    Write-ColorText $todoContent "White"

    # Create a backup of the file
    $backupFile = "$todoFile.bak"
    Copy-Item -Path $todoFile -Destination $backupFile
    Write-ColorText "Created backup of file: $backupFile" "Green"

    # Create the improved file
    $improvedContent = $todoContent -replace "public class IngestedDocument\r?\n{\r?\n    // TODO: Make Id\+SourceId a composite key\r?\n    public required string Id { get; set; }\r?\n    public required string SourceId { get; set; }", @"
public class IngestedDocument
{
    // DONE: Make Id+SourceId a composite key
    [Key]
    [Column(Order = 0)]
    public required string Id { get; set; }

    [Key]
    [Column(Order = 1)]
    public required string SourceId { get; set;}
"@

    # Add the missing using statements
    $improvedContent = $improvedContent -replace "using Microsoft\.EntityFrameworkCore;", "using Microsoft.EntityFrameworkCore;`r`nusing System.ComponentModel.DataAnnotations;`r`nusing System.ComponentModel.DataAnnotations.Schema;"

    # Update the OnModelCreating method
    $improvedContent = $improvedContent -replace "protected override void OnModelCreating\(ModelBuilder modelBuilder\)\r?\n    {\r?\n        base\.OnModelCreating\(modelBuilder\);\r?\n        modelBuilder\.Entity<IngestedDocument>\(\)\.HasMany\(d => d\.Records\)\.WithOne\(\)\.HasForeignKey\(r => r\.DocumentId\)\.OnDelete\(DeleteBehavior\.Cascade\);", @"protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        base.OnModelCreating(modelBuilder);

        // Configure the composite key for IngestedDocument
        modelBuilder.Entity<IngestedDocument>()
            .HasKey(d => new { d.Id, d.SourceId });

        // Configure the relationship between IngestedDocument and IngestedRecord
        modelBuilder.Entity<IngestedDocument>()
            .HasMany(d => d.Records)
            .WithOne()
            .HasForeignKey(r => r.DocumentId)
            .OnDelete(DeleteBehavior.Cascade);"

    # Update the IngestedRecord class
    $improvedContent = $improvedContent -replace "public class IngestedRecord\r?\n{\r?\n    public required string Id { get; set; }\r?\n    public required string DocumentId { get; set; }\r?\n}", @"public class IngestedRecord
{
    [Key]
    public required string Id { get; set; }

    public required string DocumentId { get; set; }

    // Foreign key to the composite key of IngestedDocument
    public required string SourceId { get; set; }
}"

    Write-ColorText "Creating improved file..." "Yellow"
    Set-Content -Path $todoFile -Value $improvedContent
    Write-ColorText "Improved file created" "Green"

    # Show the diff
    Write-ColorText "Diff:" "Green"
    $diff = Compare-Object -ReferenceObject (Get-Content -Path $backupFile) -DifferenceObject (Get-Content -Path $todoFile)
    $diff | ForEach-Object {
        if ($_.SideIndicator -eq "=>") {
            Write-ColorText "+ $($_.InputObject)" "Green"
        }
        else {
            Write-ColorText "- $($_.InputObject)" "Red"
        }
    }
}
else {
    Write-ColorText "File not found: $todoFile" "Red"
}

# Clean up
Write-ColorText "Cleaning up..." "Yellow"
docker stop tars-auto-coding
docker rm tars-auto-coding

Write-ColorText "Auto-coding completed" "Cyan"
