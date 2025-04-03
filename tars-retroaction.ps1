param (
    [Parameter(Mandatory=$true)]
    [string]$FilePath,
    
    [Parameter(Mandatory=$false)]
    [string]$OutputPath,
    
    [Parameter(Mandatory=$false)]
    [string]$Model = "llama3",
    
    [Parameter(Mandatory=$false)]
    [switch]$Interactive = $false,
    
    [Parameter(Mandatory=$false)]
    [switch]$ApplyImmediately = $false
)

# Function to display colored text
function Write-ColoredText {
    param (
        [string]$Text,
        [string]$Color = "White"
    )
    
    Write-Host $Text -ForegroundColor $Color
}

# Check if file exists
if (-not (Test-Path $FilePath)) {
    Write-ColoredText "Error: File not found: $FilePath" "Red"
    exit 1
}

# Determine output path
if (-not $OutputPath) {
    $directory = [System.IO.Path]::GetDirectoryName($FilePath)
    $filename = [System.IO.Path]::GetFileNameWithoutExtension($FilePath)
    $extension = [System.IO.Path]::GetExtension($FilePath)
    $OutputPath = Join-Path $directory "$($filename)_improved$extension"
}

# Step 1: Analyze the file
Write-ColoredText "`n[1/4] Analyzing file: $FilePath" "Cyan"
$analysisOutput = dotnet run --project TarsCli/TarsCli.csproj -- self-analyze --file $FilePath --model $Model

# Display analysis results
Write-ColoredText "`n[2/4] Analysis Results:" "Cyan"
$analysisOutput | ForEach-Object {
    if ($_ -match "Issue:") {
        Write-ColoredText $_ "Yellow"
    } elseif ($_ -match "Suggestion:") {
        Write-ColoredText $_ "Green"
    } else {
        Write-Host $_
    }
}

# Ask for confirmation if interactive
$shouldContinue = $true
if ($Interactive -and -not $ApplyImmediately) {
    Write-ColoredText "`nDo you want to apply the suggested improvements? (Y/N)" "Magenta"
    $response = Read-Host
    $shouldContinue = $response -eq "Y" -or $response -eq "y"
}

if (-not $shouldContinue) {
    Write-ColoredText "Operation cancelled by user." "Yellow"
    exit 0
}

# Step 3: Improve the file
Write-ColoredText "`n[3/4] Applying improvements to: $FilePath" "Cyan"
$improveOutput = dotnet run --project TarsCli/TarsCli.csproj -- self-improve --file $FilePath --output $OutputPath --model $Model

# Display improvement results
Write-ColoredText "`n[4/4] Improvement Results:" "Cyan"
$improveOutput | ForEach-Object {
    if ($_ -match "Improvement:") {
        Write-ColoredText $_ "Green"
    } elseif ($_ -match "Error:") {
        Write-ColoredText $_ "Red"
    } else {
        Write-Host $_
    }
}

# Show diff if available
if (Test-Path $OutputPath) {
    Write-ColoredText "`nImprovements applied successfully!" "Green"
    Write-ColoredText "Original file: $FilePath" "White"
    Write-ColoredText "Improved file: $OutputPath" "White"
    
    # Show a preview of the changes
    Write-ColoredText "`nChanges preview:" "Cyan"
    
    $originalContent = Get-Content $FilePath -Raw
    $improvedContent = Get-Content $OutputPath -Raw
    
    if ($originalContent -ne $improvedContent) {
        # Simple diff - show first 5 lines of each file
        Write-ColoredText "`nOriginal (first 5 lines):" "Yellow"
        Get-Content $FilePath -TotalCount 5 | ForEach-Object { Write-Host "  $_" }
        
        Write-ColoredText "`nImproved (first 5 lines):" "Green"
        Get-Content $OutputPath -TotalCount 5 | ForEach-Object { Write-Host "  $_" }
    } else {
        Write-ColoredText "No changes were made to the file." "Yellow"
    }
    
    # Ask if user wants to replace the original file
    if ($Interactive -and -not $ApplyImmediately) {
        Write-ColoredText "`nDo you want to replace the original file with the improved version? (Y/N)" "Magenta"
        $response = Read-Host
        if ($response -eq "Y" -or $response -eq "y") {
            Copy-Item -Path $OutputPath -Destination $FilePath -Force
            Write-ColoredText "Original file replaced with improved version." "Green"
        } else {
            Write-ColoredText "Original file preserved. Improved version saved at: $OutputPath" "Yellow"
        }
    } elseif ($ApplyImmediately) {
        Copy-Item -Path $OutputPath -Destination $FilePath -Force
        Write-ColoredText "Original file replaced with improved version." "Green"
    } else {
        Write-ColoredText "Original file preserved. Improved version saved at: $OutputPath" "Yellow"
    }
} else {
    Write-ColoredText "Error: Failed to generate improved file." "Red"
    exit 1
}

Write-ColoredText "`nTARS Retroaction Loop completed successfully!" "Green"
