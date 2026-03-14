param (
    [Parameter(Mandatory=$true)]
    [string]$FilePath
)

# Check if file exists
if (-not (Test-Path $FilePath)) {
    Write-Host "Error: File not found: $FilePath" -ForegroundColor Red
    exit 1
}

# Read the file content
$content = Get-Content -Path $FilePath -Raw
$improvedContent = $content

# Check file extension to determine language
$extension = [System.IO.Path]::GetExtension($FilePath).ToLower()
$improvements = 0

if ($extension -eq ".cs") {
    Write-Host "Applying C# improvements..." -ForegroundColor Green
    
    # Add using System.Linq if not present
    if (-not ($improvedContent -match "using System\.Linq;")) {
        $improvedContent = $improvedContent -replace "using System;", "using System;`r`nusing System.Linq;"
        $improvements++
        Write-Host "  - Added using System.Linq;" -ForegroundColor Green
    }
    
    # Replace string.Format with string interpolation
    $oldContent = $improvedContent
    $improvedContent = $improvedContent -replace 'string\.Format\("(.*?)"\s*,\s*(.*?)\)', '$"$1 {$2}"'
    if ($oldContent -ne $improvedContent) {
        $improvements++
        Write-Host "  - Replaced string.Format with string interpolation" -ForegroundColor Green
    }
    
    # Add null checks for reference type parameters
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
        Write-Host "  - Added null checks for string parameters" -ForegroundColor Green
    }
    
    # Add null checks for List parameters
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
        Write-Host "  - Added null checks for List parameters" -ForegroundColor Green
    }
    
    # Add divide by zero checks
    $oldContent = $improvedContent
    $improvedContent = $improvedContent -replace '(return\s+.*?)\s*\/\s*(\w+);(\s*\/\/\s*This could throw a DivideByZeroException)', 'if ($2 == 0)
        {
            throw new DivideByZeroException("Cannot divide by zero");
        }
        $1 / $2;'
    if ($oldContent -ne $improvedContent) {
        $improvements++
        Write-Host "  - Added divide by zero checks" -ForegroundColor Green
    }
    
    # Replace manual loops with LINQ for Average
    $oldContent = $improvedContent
    $improvedContent = $improvedContent -replace '(int\s+sum\s*=\s*0;\s*for\s*\(int\s+i\s*=\s*0;\s*i\s*<\s*(\w+)\.Count;\s*i\+\+\)\s*\{\s*sum\s*\+=\s*\2\[i\];\s*\}\s*return\s+sum\s*\/\s*\2\.Count;)', 'return $2.Average();'
    if ($oldContent -ne $improvedContent) {
        $improvements++
        Write-Host "  - Replaced manual loops with LINQ for Average" -ForegroundColor Green
    }
    
    # Replace manual loops with LINQ for Max
    $oldContent = $improvedContent
    $improvedContent = $improvedContent -replace '(int\s+max\s*=\s*(\w+)\[0\];\s*for\s*\(int\s+i\s*=\s*1;\s*i\s*<\s*\2\.Count;\s*i\+\+\)\s*\{\s*if\s*\(\2\[i\]\s*>\s*max\)\s*\{\s*max\s*=\s*\2\[i\];\s*\}\s*\}\s*return\s+max;)', 'return $2.Max();'
    if ($oldContent -ne $improvedContent) {
        $improvements++
        Write-Host "  - Replaced manual loops with LINQ for Max" -ForegroundColor Green
    }
}

# Check if any improvements were made
if ($improvements -eq 0) {
    Write-Host "No improvements needed for $FilePath" -ForegroundColor Yellow
    exit 0
}

# Save the improved content
$improvedContent | Set-Content -Path $FilePath -Force
Write-Host "Applied $improvements improvements to $FilePath" -ForegroundColor Green

exit 0
