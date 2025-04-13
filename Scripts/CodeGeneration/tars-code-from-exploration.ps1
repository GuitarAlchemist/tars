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
    [switch]$ApplyRetroaction = $true
)

# Function to display colored text
function Write-ColoredText {
    param (
        [string]$Text,
        [string]$Color = "White"
    )
    
    Write-Host $Text -ForegroundColor $Color
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

# Step 1: Read the exploration file
Write-ColoredText "Step 1: Reading exploration file: $ExplorationFile" "Cyan"
$explorationContent = Get-Content -Path $ExplorationFile -Raw

# Step 2: Generate code from the exploration
Write-ColoredText "Step 2: Generating $Language code from exploration..." "Cyan"

# Create a temporary file for the prompt
$promptFile = [System.IO.Path]::GetTempFileName()

# Create the prompt
$prompt = @"
You are an expert $Language developer. Your task is to generate high-quality $Language code based on the following exploration transcript.
The exploration describes requirements, ideas, and concepts that need to be implemented.

Please follow these guidelines:
1. Generate complete, working $Language code that implements the concepts described in the exploration
2. Include proper error handling, null checks, and input validation
3. Use modern language features and best practices
4. Include XML documentation comments for public members
5. Organize the code into appropriate classes, methods, and namespaces
6. Only output the code, no explanations or comments outside of the code itself

Here is the exploration transcript:

$explorationContent

Now, generate the $Language code that implements the concepts described in this exploration:
"@

# Save the prompt to the temporary file
$prompt | Set-Content -Path $promptFile -Force

# Use TARS CLI to generate the code
$generationOutput = dotnet run --project TarsCli/TarsCli.csproj -- generate --prompt-file $promptFile --model $Model

# Extract the code from the output
$codeStart = $generationOutput.IndexOf("```$Language")
if ($codeStart -eq -1) {
    $codeStart = $generationOutput.IndexOf("```")
}

$codeEnd = $generationOutput.LastIndexOf("```")

if ($codeStart -ne -1 -and $codeEnd -ne -1) {
    $codeStart = $generationOutput.IndexOf("`n", $codeStart) + 1
    $generatedCode = $generationOutput.Substring($codeStart, $codeEnd - $codeStart).Trim()
} else {
    # If no code blocks found, use the entire output
    $generatedCode = $generationOutput
}

# Save the generated code
$generatedCode | Set-Content -Path $OutputPath -Force

Write-ColoredText "Generated code saved to: $OutputPath" "Green"

# Step 3: Apply retroaction if requested
if ($ApplyRetroaction) {
    Write-ColoredText "Step 3: Applying retroaction to improve the generated code..." "Cyan"
    
    # Run the batch retroaction script
    $retroactionResult = & .\batch-tars-retroaction.ps1 -SourcePath $OutputPath -ApplyImmediately
    
    Write-ColoredText "Retroaction completed with $retroactionResult improvements" "Green"
}

# Clean up
Remove-Item -Path $promptFile -Force

Write-ColoredText "Code generation from exploration completed successfully!" "Green"
