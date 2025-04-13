param (
    [Parameter(Mandatory=$true)]
    [string]$ExplorationFile,
    
    [Parameter(Mandatory=$false)]
    [string]$OutputPath,
    
    [Parameter(Mandatory=$false)]
    [string]$Language = "csharp",
    
    [Parameter(Mandatory=$false)]
    [string]$ArchitectureStyle = "clean-architecture",
    
    [Parameter(Mandatory=$false)]
    [string]$DesignPatterns = "repository,cqrs,mediator",
    
    [Parameter(Mandatory=$false)]
    [string]$Framework = "dotnet-core",
    
    [Parameter(Mandatory=$false)]
    [string]$Model = "llama3",
    
    [Parameter(Mandatory=$false)]
    [switch]$ApplyRetroaction = $true,
    
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

# Check if exploration file exists
if (-not (Test-Path $ExplorationFile)) {
    Write-ColoredText "Error: Exploration file not found: $ExplorationFile" "Red"
    exit 1
}

# Determine output path
if (-not $OutputPath) {
    $directory = [System.IO.Path]::GetDirectoryName($ExplorationFile)
    $filename = [System.IO.Path]::GetFileNameWithoutExtension($ExplorationFile)
    $OutputPath = Join-Path $directory "$filename-generated"
}

# Create output directory if it doesn't exist
if (-not (Test-Path $OutputPath)) {
    New-Item -ItemType Directory -Path $OutputPath -Force | Out-Null
    Write-VerboseLog "Created output directory: $OutputPath"
}

# Step 1: Generate the prompt
Write-ColoredText "Step 1: Generating advanced prompt..." "Cyan"
$promptArgs = @{
    ExplorationFile = $ExplorationFile
    OutputPath = Join-Path $OutputPath "prompt"
    Language = $Language
    ArchitectureStyle = $ArchitectureStyle
    DesignPatterns = $DesignPatterns
    Framework = $Framework
    Model = $Model
    Verbose = $Verbose
}

& .\tars-advanced-prompt.ps1 @promptArgs

if ($LASTEXITCODE -ne 0) {
    Write-ColoredText "Error generating prompt" "Red"
    exit 1
}

# Step 2: Parse the generated code
Write-ColoredText "Step 2: Parsing generated code..." "Cyan"

# Read the generated README file
$readmePath = Join-Path $OutputPath "prompt\README.md"
if (-not (Test-Path $readmePath)) {
    Write-ColoredText "Error: Generated README file not found: $readmePath" "Red"
    exit 1
}

$readmeContent = Get-Content -Path $readmePath -Raw
Write-VerboseLog "README content loaded: $(($readmeContent -split "`n").Length) lines"

# Extract code blocks from the README
$codeBlocks = [regex]::Matches($readmeContent, '```(?:.*?)\r?\n(.*?)\r?\n```', [System.Text.RegularExpressions.RegexOptions]::Singleline)

if ($codeBlocks.Count -eq 0) {
    Write-ColoredText "Error: No code blocks found in the generated README" "Red"
    exit 1
}

Write-VerboseLog "Found $($codeBlocks.Count) code blocks in the README"

# Step 3: Create file structure
Write-ColoredText "Step 3: Creating file structure..." "Cyan"

# Create a directory for the code
$codePath = Join-Path $OutputPath "code"
if (-not (Test-Path $codePath)) {
    New-Item -ItemType Directory -Path $codePath -Force | Out-Null
    Write-VerboseLog "Created code directory: $codePath"
}

# Parse file paths and content from the code blocks
$filePattern = '// File: (.*?)\r?\n(.*?)(?=\r?\n// File:|$)'
$fileMatches = [regex]::Matches($codeBlocks[0].Groups[1].Value, $filePattern, [System.Text.RegularExpressions.RegexOptions]::Singleline)

if ($fileMatches.Count -eq 0) {
    # If no file markers are found, try to infer file structure from the code
    Write-VerboseLog "No file markers found, inferring file structure from code"
    
    # For C#, look for namespace declarations
    if ($Language -eq "csharp") {
        $namespacePattern = 'namespace\s+([\w\.]+)'
        $namespaceMatches = [regex]::Matches($codeBlocks[0].Groups[1].Value, $namespacePattern)
        
        if ($namespaceMatches.Count -gt 0) {
            $namespace = $namespaceMatches[0].Groups[1].Value
            $className = "Program"
            
            # Look for class declarations
            $classPattern = 'class\s+(\w+)'
            $classMatches = [regex]::Matches($codeBlocks[0].Groups[1].Value, $classPattern)
            
            if ($classMatches.Count -gt 0) {
                $className = $classMatches[0].Groups[1].Value
            }
            
            # Create a file for the code
            $filePath = Join-Path $codePath "$className.$Language"
            $codeBlocks[0].Groups[1].Value | Set-Content -Path $filePath -Force
            
            Write-ColoredText "Created file: $filePath" "Green"
        }
        else {
            # If no namespace is found, create a single file
            $filePath = Join-Path $codePath "Program.$Language"
            $codeBlocks[0].Groups[1].Value | Set-Content -Path $filePath -Force
            
            Write-ColoredText "Created file: $filePath" "Green"
        }
    }
    else {
        # For other languages, create a single file
        $filePath = Join-Path $codePath "main.$Language"
        $codeBlocks[0].Groups[1].Value | Set-Content -Path $filePath -Force
        
        Write-ColoredText "Created file: $filePath" "Green"
    }
}
else {
    # Create files based on the file markers
    foreach ($fileMatch in $fileMatches) {
        $filePath = $fileMatch.Groups[1].Value.Trim()
        $fileContent = $fileMatch.Groups[2].Value
        
        # Create the directory structure
        $fileDir = [System.IO.Path]::GetDirectoryName(Join-Path $codePath $filePath)
        if (-not (Test-Path $fileDir)) {
            New-Item -ItemType Directory -Path $fileDir -Force | Out-Null
            Write-VerboseLog "Created directory: $fileDir"
        }
        
        # Create the file
        $fileContent | Set-Content -Path (Join-Path $codePath $filePath) -Force
        
        Write-ColoredText "Created file: $filePath" "Green"
    }
}

# Step 4: Apply retroaction if requested
if ($ApplyRetroaction) {
    Write-ColoredText "Step 4: Applying retroaction to improve the generated code..." "Cyan"
    
    # Find all code files
    $codeFiles = Get-ChildItem -Path $codePath -Recurse -File | Where-Object { $_.Extension -match "\.(cs|fs|ts|js|py)$" }
    
    foreach ($file in $codeFiles) {
        Write-ColoredText "Applying retroaction to: $($file.FullName)" "Cyan"
        
        # Determine the language
        $fileLanguage = switch ($file.Extension) {
            ".cs" { "csharp" }
            ".fs" { "fsharp" }
            ".ts" { "typescript" }
            ".js" { "javascript" }
            ".py" { "python" }
            default { $Language }
        }
        
        # Apply retroaction
        & .\simple-retroaction.ps1 -FilePath $file.FullName
        
        if ($LASTEXITCODE -ne 0) {
            Write-ColoredText "Warning: Retroaction failed for file: $($file.FullName)" "Yellow"
        }
        else {
            Write-ColoredText "Retroaction applied to: $($file.FullName)" "Green"
        }
    }
}

# Step 5: Create a project file if needed
if ($Language -eq "csharp" -or $Language -eq "fsharp") {
    Write-ColoredText "Step 5: Creating project file..." "Cyan"
    
    $projectType = if ($Language -eq "csharp") { "csproj" } else { "fsproj" }
    $projectFile = Join-Path $codePath "Project.$projectType"
    
    if (-not (Test-Path $projectFile)) {
        # Determine the project SDK
        $sdk = "Microsoft.NET.Sdk"
        
        if ($Framework -eq "dotnet-core" -and (Get-ChildItem -Path $codePath -Recurse -File | Where-Object { $_.Name -match "Controller\.cs$" })) {
            $sdk = "Microsoft.NET.Sdk.Web"
        }
        
        # Create a basic project file
        @"
<Project Sdk="$sdk">

  <PropertyGroup>
    <TargetFramework>net6.0</TargetFramework>
    <Nullable>enable</Nullable>
    <ImplicitUsings>enable</ImplicitUsings>
  </PropertyGroup>

</Project>
"@ | Set-Content -Path $projectFile -Force
        
        Write-ColoredText "Created project file: $projectFile" "Green"
    }
}

# Step 6: Create a README file
Write-ColoredText "Step 6: Creating README file..." "Cyan"

$readmeFile = Join-Path $codePath "README.md"
@"
# Generated Code

This code was generated based on the exploration file: $ExplorationFile

## Generation Parameters
- Language: $Language
- Architecture: $ArchitectureStyle
- Design Patterns: $DesignPatterns
- Framework: $Framework
- Model: $Model

## Project Structure

The following files were generated:

$(Get-ChildItem -Path $codePath -Recurse -File | ForEach-Object { "- $($_.FullName.Substring($codePath.Length + 1))" } | Out-String)

## How to Run

$(
    if ($Language -eq "csharp" -or $Language -eq "fsharp") {
        "1. Navigate to the project directory: `cd $codePath`"
        "2. Restore dependencies: `dotnet restore`"
        "3. Build the project: `dotnet build`"
        "4. Run the project: `dotnet run`"
    }
    elseif ($Language -eq "typescript") {
        "1. Navigate to the project directory: `cd $codePath`"
        "2. Install dependencies: `npm install`"
        "3. Build the project: `npm run build`"
        "4. Run the project: `npm start`"
    }
    elseif ($Language -eq "python") {
        "1. Navigate to the project directory: `cd $codePath`"
        "2. Install dependencies: `pip install -r requirements.txt`"
        "3. Run the project: `python main.py`"
    }
    else {
        "Please refer to the language-specific documentation for running the code."
    }
)

## Next Steps
1. Review the generated code
2. Make any necessary adjustments
3. Run the code to verify functionality
4. Write additional tests as needed
"@ | Set-Content -Path $readmeFile -Force

Write-ColoredText "Created README file: $readmeFile" "Green"

Write-ColoredText "Code generation completed successfully!" "Green"
Write-ColoredText "Generated code is available at: $codePath" "Green"
