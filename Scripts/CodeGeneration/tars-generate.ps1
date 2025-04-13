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
    [switch]$ApplyRetroaction = $true,
    
    [Parameter(Mandatory=$false)]
    [string]$ProjectContext = "",
    
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
function Write-Verbose {
    param (
        [string]$Text
    )
    
    if ($Verbose) {
        Write-ColoredText "VERBOSE: $Text" "DarkGray"
    }
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
Write-Verbose "Exploration content loaded: $(($explorationContent -split "`n").Length) lines"

# Step 2: Gather project context if specified
$contextContent = ""
if ($ProjectContext -and (Test-Path $ProjectContext)) {
    Write-ColoredText "Step 2: Gathering project context from: $ProjectContext" "Cyan"
    
    if (Test-Path -Path $ProjectContext -PathType Container) {
        # It's a directory, get all code files
        $contextFiles = Get-ChildItem -Path $ProjectContext -Recurse -File | Where-Object {
            $_.Extension -in @(".cs", ".fs", ".fsx", ".csx")
        } | Select-Object -First 10  # Limit to 10 files to avoid context overflow
        
        foreach ($file in $contextFiles) {
            $fileContent = Get-Content -Path $file.FullName -Raw
            $contextContent += "// File: $($file.FullName)`n$fileContent`n`n"
        }
        
        Write-Verbose "Gathered context from $($contextFiles.Count) files"
    } else {
        # It's a single file
        $contextContent = Get-Content -Path $ProjectContext -Raw
        Write-Verbose "Gathered context from single file: $ProjectContext"
    }
}

# Step 3: Generate code from the exploration
Write-ColoredText "Step 3: Generating $Language code from exploration..." "Cyan"

# Create a temporary file for the prompt
$promptFile = [System.IO.Path]::GetTempFileName()

# Create the prompt with enhanced engineering
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

"@

# Add language-specific guidance
if ($Language -eq "csharp") {
    $prompt += @"
For C# code:
- Use C# 10.0 features where appropriate
- Follow Microsoft's C# coding conventions
- Use LINQ for collection operations
- Use async/await for asynchronous operations
- Use nullable reference types
- Use pattern matching where appropriate
- Prefer expression-bodied members for simple methods
- Use string interpolation instead of string.Format

"@
} elseif ($Language -eq "fsharp") {
    $prompt += @"
For F# code:
- Use F# 6.0 features where appropriate
- Follow F# coding conventions
- Use functional programming patterns
- Use discriminated unions for modeling domain concepts
- Use computation expressions for complex workflows
- Use pattern matching extensively
- Use railway-oriented programming for error handling
- Use type providers where appropriate

"@
}

# Add project context if available
if ($contextContent) {
    $prompt += @"
Here is some context from the existing project that may be helpful:

$contextContent

"@
}

# Add the exploration content
$prompt += @"
Here is the exploration transcript:

$explorationContent

Now, generate the $Language code that implements the concepts described in this exploration:
"@

# Save the prompt to the temporary file
$prompt | Set-Content -Path $promptFile -Force
Write-Verbose "Prompt saved to temporary file: $promptFile"

# Use TARS CLI to generate the code
Write-Verbose "Calling TARS CLI with model: $Model"
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
    Write-Verbose "Successfully extracted code from output"
} else {
    # If no code blocks found, use the entire output
    $generatedCode = $generationOutput
    Write-Verbose "No code blocks found, using entire output"
}

# Save the generated code
$generatedCode | Set-Content -Path $OutputPath -Force
Write-ColoredText "Generated code saved to: $OutputPath" "Green"

# Step 4: Apply retroaction if requested
if ($ApplyRetroaction) {
    Write-ColoredText "Step 4: Applying retroaction to improve the generated code..." "Cyan"
    
    # Run the retroaction script
    & .\simple-retroaction.ps1 -FilePath $OutputPath
    
    Write-ColoredText "Retroaction completed" "Green"
}

# Clean up
Remove-Item -Path $promptFile -Force
Write-Verbose "Temporary prompt file removed"

Write-ColoredText "Code generation from exploration completed successfully!" "Green"
