param (
    [Parameter(Mandatory=$true)]
    [string]$FilePath,
    
    [Parameter(Mandatory=$false)]
    [string]$OutputPath
)

# Check if file exists
if (-not (Test-Path $FilePath)) {
    Write-Host "Error: File not found: $FilePath" -ForegroundColor Red
    exit 1
}

# Determine output path
if (-not $OutputPath) {
    $directory = [System.IO.Path]::GetDirectoryName($FilePath)
    $filename = [System.IO.Path]::GetFileNameWithoutExtension($FilePath)
    $extension = [System.IO.Path]::GetExtension($FilePath)
    $OutputPath = Join-Path $directory "$($filename)_improved$extension"
}

# Create a temporary directory for our work
$tempDir = Join-Path $env:TEMP "TarsRetroaction_$(Get-Random)"
New-Item -ItemType Directory -Path $tempDir -Force | Out-Null

try {
    # Step 1: Copy the original file to analyze
    $tempFile = Join-Path $tempDir (Split-Path $FilePath -Leaf)
    Copy-Item -Path $FilePath -Destination $tempFile -Force
    
    Write-Host "Step 1: Analyzing file: $FilePath..." -ForegroundColor Cyan
    
    # Step 2: Manually analyze the file and create an improved version
    $content = Get-Content -Path $FilePath -Raw
    $improvedContent = $content
    
    # Check file extension to determine language
    $extension = [System.IO.Path]::GetExtension($FilePath).ToLower()
    
    if ($extension -eq ".cs") {
        Write-Host "Detected C# file, applying C# improvements..." -ForegroundColor Green
        
        # Add using System.Linq if not present
        if (-not ($improvedContent -match "using System\.Linq;")) {
            $improvedContent = $improvedContent -replace "using System;", "using System;`r`nusing System.Linq;"
        }
        
        # Replace string.Format with string interpolation
        $improvedContent = $improvedContent -replace 'string\.Format\("(.*?)"\s*,\s*(.*?)\)', '$"$1 {$2}"'
        $improvedContent = $improvedContent -replace 'string\.Format\("(.*?)"\s*,\s*(.*?),\s*(.*?)\)', '$"$1 {$2} {$3}"'
        
        # Add null checks for reference type parameters
        $improvedContent = $improvedContent -replace '(public\s+\w+\s+\w+\(.*?string\s+(\w+).*?\))\s*\{', '$1
        {
            if ($2 == null)
            {
                throw new ArgumentNullException(nameof($2));
            }
'
        
        # Add null checks for List parameters
        $improvedContent = $improvedContent -replace '(public\s+\w+\s+\w+\(.*?List<.*?>\s+(\w+).*?\))\s*\{', '$1
        {
            if ($2 == null)
            {
                throw new ArgumentNullException(nameof($2));
            }
'
        
        # Add divide by zero checks
        $improvedContent = $improvedContent -replace '(return\s+.*?)\s*\/\s*(\w+);(\s*\/\/\s*This could throw a DivideByZeroException)', 'if ($2 == 0)
            {
                throw new DivideByZeroException("Cannot divide by zero");
            }
            $1 / $2;'
        
        # Replace manual loops with LINQ for Average
        $improvedContent = $improvedContent -replace '(int\s+sum\s*=\s*0;\s*for\s*\(int\s+i\s*=\s*0;\s*i\s*<\s*(\w+)\.Count;\s*i\+\+\)\s*\{\s*sum\s*\+=\s*\2\[i\];\s*\}\s*return\s+sum\s*\/\s*\2\.Count;)', 'return $2.Average();'
        
        # Replace manual loops with LINQ for Max
        $improvedContent = $improvedContent -replace '(int\s+max\s*=\s*(\w+)\[0\];\s*for\s*\(int\s+i\s*=\s*1;\s*i\s*<\s*\2\.Count;\s*i\+\+\)\s*\{\s*if\s*\(\2\[i\]\s*>\s*max\)\s*\{\s*max\s*=\s*\2\[i\];\s*\}\s*\}\s*return\s+max;)', 'return $2.Max();'
    }
    elseif ($extension -eq ".fs") {
        Write-Host "Detected F# file, applying F# improvements..." -ForegroundColor Green
        
        # Add open System.Linq if not present
        if (-not ($improvedContent -match "open System\.Linq")) {
            $improvedContent = $improvedContent -replace "open System", "open System`r`nopen System.Linq"
        }
        
        # Replace imperative loops with functional approaches
        $improvedContent = $improvedContent -replace "for i in 0 .. (.*?)\.Length - 1 do", "Array.iteri (fun i ->"
        $improvedContent = $improvedContent -replace "for i in 0 .. (.*?)\.Count - 1 do", "List.iteri (fun i ->"
    }
    else {
        Write-Host "Unsupported file type: $extension" -ForegroundColor Yellow
    }
    
    # Save the improved content
    $improvedContent | Set-Content -Path $OutputPath -Force
    
    # Step 3: Show the differences
    Write-Host "`nStep 3: Showing differences between original and improved files..." -ForegroundColor Cyan
    
    Write-Host "`nOriginal file (first 10 lines):" -ForegroundColor Yellow
    Get-Content $FilePath -TotalCount 10 | ForEach-Object { Write-Host "  $_" }
    
    Write-Host "`nImproved file (first 10 lines):" -ForegroundColor Green
    Get-Content $OutputPath -TotalCount 10 | ForEach-Object { Write-Host "  $_" }
    
    Write-Host "`nImproved file saved to: $OutputPath" -ForegroundColor Green
    
    # Step 4: Ask if user wants to replace the original file
    Write-Host "`nDo you want to replace the original file with the improved version? (Y/N)" -ForegroundColor Magenta
    $response = Read-Host
    if ($response -eq "Y" -or $response -eq "y") {
        Copy-Item -Path $OutputPath -Destination $FilePath -Force
        Write-Host "Original file replaced with improved version." -ForegroundColor Green
    } else {
        Write-Host "Original file preserved. Improved version saved at: $OutputPath" -ForegroundColor Yellow
    }
}
finally {
    # Clean up
    if (Test-Path $tempDir) {
        Remove-Item -Path $tempDir -Recurse -Force
    }
}

Write-Host "`nTARS Retroaction completed successfully!" -ForegroundColor Green
