param (
    [Parameter(Mandatory=$true)]
    [string]$ExplorationFile,
    
    [Parameter(Mandatory=$false)]
    [string]$OutputPath,
    
    [Parameter(Mandatory=$false)]
    [string]$Language = "csharp",
    
    [Parameter(Mandatory=$false)]
    [string]$Model = "llama3",
    
    [Parameter(Mandatory=$false)]
    [string]$ProjectContext = "",
    
    [Parameter(Mandatory=$false)]
    [switch]$SkipRetroaction = $false,
    
    [Parameter(Mandatory=$false)]
    [switch]$CommitToGit = $false,
    
    [Parameter(Mandatory=$false)]
    [string]$CommitMessage = "",
    
    [Parameter(Mandatory=$false)]
    [switch]$Verbose = $false
)

# Function to display colored text
function Write-ColoredText {
    param (
        [string]$Text,
        [string]$Color = "White"
    )
    
    Write-Host $Text -ForegroundColor $Color
}

# Function to log verbose information
function Write-VerboseLog {
    param (
        [string]$Text
    )
    
    if ($Verbose) {
        Write-ColoredText "VERBOSE: $Text" "DarkGray"
    }
}

# Function to check if a command exists
function Test-Command {
    param (
        [string]$Command
    )
    
    $null -ne (Get-Command $Command -ErrorAction SilentlyContinue)
}

# Check if exploration file exists
if (-not (Test-Path $ExplorationFile)) {
    Write-ColoredText "Error: Exploration file not found: $ExplorationFile" "Red"
    exit 1
}

# Determine output path
if (-not $OutputPath) {
    $directory = [System.IO.Path]::GetDirectoryName($ExplorationFile)
    $filename = [System.IO.Path]::GetFileNameWithoutExtension($ExplorationFile)
    
    if ($Language -eq "csharp") {
        $OutputPath = Join-Path $directory "$filename.cs"
    } elseif ($Language -eq "fsharp") {
        $OutputPath = Join-Path $directory "$filename.fs"
    } else {
        $OutputPath = Join-Path $directory "$filename.txt"
    }
}

# Create a log file
$logFile = "$ExplorationFile.log"
"" | Set-Content -Path $logFile -Force
function Write-Log {
    param (
        [string]$Message
    )
    
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$timestamp - $Message" | Add-Content -Path $logFile
}

Write-Log "Starting TARS workflow for $ExplorationFile"
Write-Log "Output path: $OutputPath"
Write-Log "Language: $Language"
Write-Log "Model: $Model"

# Step 1: Generate code from exploration
Write-ColoredText "Step 1: Generating code from exploration..." "Cyan"
Write-Log "Generating code from exploration"

$generateParams = @{
    ExplorationFile = $ExplorationFile
    OutputPath = $OutputPath
    Language = $Language
    Model = $Model
    ApplyRetroaction = $false  # We'll do this separately
    Verbose = $Verbose
}

if ($ProjectContext) {
    $generateParams.ProjectContext = $ProjectContext
}

& .\tars-generate.ps1 @generateParams

if (-not (Test-Path $OutputPath)) {
    Write-ColoredText "Error: Code generation failed. Output file not created." "Red"
    Write-Log "Error: Code generation failed. Output file not created."
    exit 1
}

Write-Log "Code generation completed. Output saved to: $OutputPath"

# Step 2: Apply retroaction if not skipped
if (-not $SkipRetroaction) {
    Write-ColoredText "Step 2: Applying retroaction to improve the generated code..." "Cyan"
    Write-Log "Applying retroaction"
    
    & .\advanced-retroaction.ps1 -FilePath $OutputPath -CreateBackup -Verbose:$Verbose
    
    Write-Log "Retroaction completed"
} else {
    Write-ColoredText "Step 2: Skipping retroaction (as requested)" "Yellow"
    Write-Log "Retroaction skipped"
}

# Step 3: Commit to Git if requested
if ($CommitToGit) {
    Write-ColoredText "Step 3: Committing changes to Git..." "Cyan"
    
    # Check if git is available
    if (-not (Test-Command "git")) {
        Write-ColoredText "Error: Git command not found. Cannot commit changes." "Red"
        Write-Log "Error: Git command not found. Cannot commit changes."
    } else {
        # Check if the file is in a git repository
        $gitStatus = git status --porcelain $OutputPath 2>&1
        
        if ($LASTEXITCODE -ne 0) {
            Write-ColoredText "Error: $OutputPath is not in a Git repository." "Red"
            Write-Log "Error: $OutputPath is not in a Git repository."
        } else {
            # Add the file to git
            git add $OutputPath
            
            # Create commit message if not provided
            if (-not $CommitMessage) {
                $CommitMessage = "Generated code from exploration: $ExplorationFile"
            }
            
            # Commit the changes
            git commit -m $CommitMessage
            
            Write-ColoredText "Changes committed to Git with message: $CommitMessage" "Green"
            Write-Log "Changes committed to Git with message: $CommitMessage"
        }
    }
} else {
    Write-ColoredText "Step 3: Skipping Git commit (as requested)" "Yellow"
    Write-Log "Git commit skipped"
}

# Step 4: Display summary
Write-ColoredText "`nTARS Workflow Summary:" "Magenta"
Write-ColoredText "------------------------" "Magenta"
Write-ColoredText "Exploration file: $ExplorationFile" "White"
Write-ColoredText "Generated code: $OutputPath" "White"
Write-ColoredText "Language: $Language" "White"
Write-ColoredText "Model used: $Model" "White"
Write-ColoredText "Retroaction applied: $(-not $SkipRetroaction)" "White"
Write-ColoredText "Git commit: $CommitToGit" "White"
Write-ColoredText "Log file: $logFile" "White"
Write-ColoredText "------------------------" "Magenta"

Write-Log "TARS workflow completed successfully"
Write-ColoredText "`nTARS workflow completed successfully!" "Green"
