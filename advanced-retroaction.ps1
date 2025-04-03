param (
    [Parameter(Mandatory=$true)]
    [string]$FilePath,
    
    [Parameter(Mandatory=$false)]
    [switch]$CreateBackup = $true,
    
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

# Check if file exists
if (-not (Test-Path $FilePath)) {
    Write-ColoredText "Error: File not found: $FilePath" "Red"
    exit 1
}

# Create a backup if requested
if ($CreateBackup) {
    $backupPath = "$FilePath.bak"
    Copy-Item -Path $FilePath -Destination $backupPath -Force
    Write-VerboseLog "Created backup at: $backupPath"
}

# Read the file content
$content = Get-Content -Path $FilePath -Raw
$improvedContent = $content

# Check file extension to determine language
$extension = [System.IO.Path]::GetExtension($FilePath).ToLower()
$improvements = 0

if ($extension -eq ".cs") {
    Write-ColoredText "Applying C# improvements..." "Green"
    
    # 1. Add using statements
    $usingsToAdd = @(
        "System.Linq",
        "System.Collections.Generic",
        "System.Threading.Tasks"
    )
    
    foreach ($using in $usingsToAdd) {
        if (-not ($improvedContent -match "using\s+$using;")) {
            $oldContent = $improvedContent
            $improvedContent = $improvedContent -replace "using System;", "using System;`r`nusing $using;"
            if ($oldContent -ne $improvedContent) {
                $improvements++
                Write-ColoredText "  - Added using $using;" "Green"
            }
        }
    }
    
    # 2. Replace string.Format with string interpolation
    $oldContent = $improvedContent
    $improvedContent = $improvedContent -replace 'string\.Format\("(.*?)"\s*,\s*(.*?)\)', '$"$1 {$2}"'
    $improvedContent = $improvedContent -replace 'string\.Format\("(.*?)"\s*,\s*(.*?),\s*(.*?)\)', '$"$1 {$2} {$3}"'
    if ($oldContent -ne $improvedContent) {
        $improvements++
        Write-ColoredText "  - Replaced string.Format with string interpolation" "Green"
    }
    
    # 3. Add null checks for reference type parameters
    $oldContent = $improvedContent
    $improvedContent = $improvedContent -replace '(public\s+\w+\s+\w+\(.*?string\s+(\w+).*?\))\s*\{', '$1
    {
        if ($2 == null)
        {
            throw new ArgumentNullException(nameof($2));
        }
'
    if ($oldContent -ne $improvedContent) {
        $improvements++
        Write-ColoredText "  - Added null checks for string parameters" "Green"
    }
    
    # 4. Add null checks for List parameters
    $oldContent = $improvedContent
    $improvedContent = $improvedContent -replace '(public\s+\w+\s+\w+\(.*?List<.*?>\s+(\w+).*?\))\s*\{', '$1
    {
        if ($2 == null)
        {
            throw new ArgumentNullException(nameof($2));
        }
'
    if ($oldContent -ne $improvedContent) {
        $improvements++
        Write-ColoredText "  - Added null checks for List parameters" "Green"
    }
    
    # 5. Add null checks for IEnumerable parameters
    $oldContent = $improvedContent
    $improvedContent = $improvedContent -replace '(public\s+\w+\s+\w+\(.*?IEnumerable<.*?>\s+(\w+).*?\))\s*\{', '$1
    {
        if ($2 == null)
        {
            throw new ArgumentNullException(nameof($2));
        }
'
    if ($oldContent -ne $improvedContent) {
        $improvements++
        Write-ColoredText "  - Added null checks for IEnumerable parameters" "Green"
    }
    
    # 6. Add divide by zero checks
    $oldContent = $improvedContent
    $improvedContent = $improvedContent -replace '(return\s+.*?)\s*\/\s*(\w+);(\s*\/\/\s*This could throw a DivideByZeroException)', 'if ($2 == 0)
        {
            throw new DivideByZeroException("Cannot divide by zero");
        }
        $1 / $2;'
    if ($oldContent -ne $improvedContent) {
        $improvements++
        Write-ColoredText "  - Added divide by zero checks" "Green"
    }
    
    # 7. Replace manual loops with LINQ for Average
    $oldContent = $improvedContent
    $improvedContent = $improvedContent -replace '(int\s+sum\s*=\s*0;\s*for\s*\(int\s+i\s*=\s*0;\s*i\s*<\s*(\w+)\.Count;\s*i\+\+\)\s*\{\s*sum\s*\+=\s*\2\[i\];\s*\}\s*return\s+sum\s*\/\s*\2\.Count;)', 'return $2.Average();'
    if ($oldContent -ne $improvedContent) {
        $improvements++
        Write-ColoredText "  - Replaced manual loops with LINQ for Average" "Green"
    }
    
    # 8. Replace manual loops with LINQ for Max
    $oldContent = $improvedContent
    $improvedContent = $improvedContent -replace '(int\s+max\s*=\s*(\w+)\[0\];\s*for\s*\(int\s+i\s*=\s*1;\s*i\s*<\s*\2\.Count;\s*i\+\+\)\s*\{\s*if\s*\(\2\[i\]\s*>\s*max\)\s*\{\s*max\s*=\s*\2\[i\];\s*\}\s*\}\s*return\s+max;)', 'return $2.Max();'
    if ($oldContent -ne $improvedContent) {
        $improvements++
        Write-ColoredText "  - Replaced manual loops with LINQ for Max" "Green"
    }
    
    # 9. Replace manual loops with LINQ for Sum
    $oldContent = $improvedContent
    $improvedContent = $improvedContent -replace '(int\s+sum\s*=\s*0;\s*foreach\s*\(\s*var\s+(\w+)\s+in\s+(\w+)\s*\)\s*\{\s*sum\s*\+=\s*\2;\s*\}\s*return\s+sum;)', 'return $3.Sum();'
    if ($oldContent -ne $improvedContent) {
        $improvements++
        Write-ColoredText "  - Replaced manual loops with LINQ for Sum" "Green"
    }
    
    # 10. Replace manual loops with LINQ for Any
    $oldContent = $improvedContent
    $improvedContent = $improvedContent -replace '(foreach\s*\(\s*var\s+(\w+)\s+in\s+(\w+)\s*\)\s*\{\s*if\s*\(\s*\2\s*==\s*(.*?)\s*\)\s*\{\s*return\s+true;\s*\}\s*\}\s*return\s+false;)', 'return $3.Any(item => item == $4);'
    if ($oldContent -ne $improvedContent) {
        $improvements++
        Write-ColoredText "  - Replaced manual loops with LINQ for Any" "Green"
    }
    
    # 11. Replace manual loops with LINQ for All
    $oldContent = $improvedContent
    $improvedContent = $improvedContent -replace '(foreach\s*\(\s*var\s+(\w+)\s+in\s+(\w+)\s*\)\s*\{\s*if\s*\(\s*!\s*\(.*?\)\s*\)\s*\{\s*return\s+false;\s*\}\s*\}\s*return\s+true;)', 'return $3.All(item => $4);'
    if ($oldContent -ne $improvedContent) {
        $improvements++
        Write-ColoredText "  - Replaced manual loops with LINQ for All" "Green"
    }
    
    # 12. Convert properties to expression-bodied members
    $oldContent = $improvedContent
    $improvedContent = $improvedContent -replace '(public\s+\w+\s+\w+\s*\{\s*get\s*\{\s*return\s+(.*?);\s*\}\s*\})', 'public $1 => $2;'
    if ($oldContent -ne $improvedContent) {
        $improvements++
        Write-ColoredText "  - Converted properties to expression-bodied members" "Green"
    }
    
    # 13. Convert simple methods to expression-bodied members
    $oldContent = $improvedContent
    $improvedContent = $improvedContent -replace '(public\s+\w+\s+\w+\s*\(.*?\)\s*\{\s*return\s+(.*?);\s*\})', 'public $1 => $2;'
    if ($oldContent -ne $improvedContent) {
        $improvements++
        Write-ColoredText "  - Converted simple methods to expression-bodied members" "Green"
    }
    
    # 14. Add missing XML documentation
    $oldContent = $improvedContent
    $improvedContent = $improvedContent -replace '(public\s+class\s+(\w+)(?!\s*:))', '/// <summary>
/// Represents a $2.
/// </summary>
$1'
    if ($oldContent -ne $improvedContent) {
        $improvements++
        Write-ColoredText "  - Added missing XML documentation for classes" "Green"
    }
    
    # 15. Add missing XML documentation for public methods
    $oldContent = $improvedContent
    $improvedContent = $improvedContent -replace '(?<!\/\/\/.*\n\s*)(?<!\/\*.*\n\s*)(public\s+\w+\s+(\w+)\s*\((.*?)\)\s*(?:=>|{))', '/// <summary>
    /// $2 operation.
    /// </summary>
    $1'
    if ($oldContent -ne $improvedContent) {
        $improvements++
        Write-ColoredText "  - Added missing XML documentation for public methods" "Green"
    }
}
elseif ($extension -eq ".fs") {
    Write-ColoredText "Applying F# improvements..." "Green"
    
    # 1. Add open statements
    $opensToAdd = @(
        "System.Linq",
        "System.Collections.Generic"
    )
    
    foreach ($open in $opensToAdd) {
        if (-not ($improvedContent -match "open\s+$open")) {
            $oldContent = $improvedContent
            $improvedContent = $improvedContent -replace "open System", "open System`r`nopen $open"
            if ($oldContent -ne $improvedContent) {
                $improvements++
                Write-ColoredText "  - Added open $open" "Green"
            }
        }
    }
    
    # 2. Replace imperative loops with functional approaches
    $oldContent = $improvedContent
    $improvedContent = $improvedContent -replace "for i in 0 .. (.*?)\.Length - 1 do", "Array.iteri (fun i ->"
    if ($oldContent -ne $improvedContent) {
        $improvements++
        Write-ColoredText "  - Replaced imperative loops with functional approaches" "Green"
    }
    
    # 3. Replace mutable variables with immutable when possible
    $oldContent = $improvedContent
    $improvedContent = $improvedContent -replace "let mutable (\w+) = (.*?)\s+(\w+) <- ", "let $1 = $2\r\n    let $1 = "
    if ($oldContent -ne $improvedContent) {
        $improvements++
        Write-ColoredText "  - Replaced mutable variables with immutable when possible" "Green"
    }
    
    # 4. Add XML documentation
    $oldContent = $improvedContent
    $improvedContent = $improvedContent -replace "let (\w+) (.*?) =", "/// <summary>\r\n/// $1 function.\r\n/// </summary>\r\nlet $1 $2 ="
    if ($oldContent -ne $improvedContent) {
        $improvements++
        Write-ColoredText "  - Added XML documentation" "Green"
    }
}
else {
    Write-ColoredText "Unsupported file type: $extension" "Yellow"
    exit 1
}

# Check if any improvements were made
if ($improvements -eq 0) {
    Write-ColoredText "No improvements needed for $FilePath" "Yellow"
    exit 0
}

# Save the improved content
$improvedContent | Set-Content -Path $FilePath -Force
Write-ColoredText "Applied $improvements improvements to $FilePath" "Green"

exit 0
