param (
    [Parameter(Mandatory=$false)]
    [string]$SourcePath = ".",
    
    [Parameter(Mandatory=$false)]
    [string]$FilePattern = "*.cs",
    
    [Parameter(Mandatory=$false)]
    [switch]$Recursive = $false,
    
    [Parameter(Mandatory=$false)]
    [switch]$ApplyImmediately = $true,
    
    [Parameter(Mandatory=$false)]
    [string]$LogFile = "tars-retroaction-log.txt"
)

# Function to log messages
function Write-Log {
    param (
        [string]$Message,
        [string]$Color = "White"
    )
    
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] $Message"
    
    # Write to console
    Write-Host $logMessage -ForegroundColor $Color
    
    # Write to log file
    Add-Content -Path $LogFile -Value $logMessage
}

# Function to improve a single file
function Improve-File {
    param (
        [string]$FilePath
    )
    
    Write-Log "Processing file: $FilePath" "Cyan"
    
    # Check if file exists
    if (-not (Test-Path $FilePath)) {
        Write-Log "Error: File not found: $FilePath" "Red"
        return $false
    }
    
    # Create a backup of the original file
    $backupPath = "$FilePath.bak"
    Copy-Item -Path $FilePath -Destination $backupPath -Force
    
    try {
        # Read the file content
        $content = Get-Content -Path $FilePath -Raw
        $improvedContent = $content
        
        # Check file extension to determine language
        $extension = [System.IO.Path]::GetExtension($FilePath).ToLower()
        $improvements = 0
        
        if ($extension -eq ".cs") {
            Write-Log "Applying C# improvements..." "Green"
            
            # Add using System.Linq if not present
            if (-not ($improvedContent -match "using System\.Linq;")) {
                $improvedContent = $improvedContent -replace "using System;", "using System;`r`nusing System.Linq;"
                $improvements++
                Write-Log "  - Added using System.Linq;" "Green"
            }
            
            # Replace string.Format with string interpolation
            $oldContent = $improvedContent
            $improvedContent = $improvedContent -replace 'string\.Format\("(.*?)"\s*,\s*(.*?)\)', '$"$1 {$2}"'
            if ($oldContent -ne $improvedContent) {
                $improvements++
                Write-Log "  - Replaced string.Format with string interpolation" "Green"
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
                Write-Log "  - Added null checks for string parameters" "Green"
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
                Write-Log "  - Added null checks for List parameters" "Green"
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
                Write-Log "  - Added divide by zero checks" "Green"
            }
            
            # Replace manual loops with LINQ for Average
            $oldContent = $improvedContent
            $improvedContent = $improvedContent -replace '(int\s+sum\s*=\s*0;\s*for\s*\(int\s+i\s*=\s*0;\s*i\s*<\s*(\w+)\.Count;\s*i\+\+\)\s*\{\s*sum\s*\+=\s*\2\[i\];\s*\}\s*return\s+sum\s*\/\s*\2\.Count;)', 'return $2.Average();'
            if ($oldContent -ne $improvedContent) {
                $improvements++
                Write-Log "  - Replaced manual loops with LINQ for Average" "Green"
            }
            
            # Replace manual loops with LINQ for Max
            $oldContent = $improvedContent
            $improvedContent = $improvedContent -replace '(int\s+max\s*=\s*(\w+)\[0\];\s*for\s*\(int\s+i\s*=\s*1;\s*i\s*<\s*\2\.Count;\s*i\+\+\)\s*\{\s*if\s*\(\2\[i\]\s*>\s*max\)\s*\{\s*max\s*=\s*\2\[i\];\s*\}\s*\}\s*return\s+max;)', 'return $2.Max();'
            if ($oldContent -ne $improvedContent) {
                $improvements++
                Write-Log "  - Replaced manual loops with LINQ for Max" "Green"
            }
            
            # Fix string interpolation issues
            $oldContent = $improvedContent
            $improvedContent = $improvedContent -replace '\$"\{(\d+)\}:\s*\{(\d+)\}\s*\{(.*?),\s*(.*?)\}"', '$"$3: $4"'
            if ($oldContent -ne $improvedContent) {
                $improvements++
                Write-Log "  - Fixed string interpolation issues" "Green"
            }
        }
        elseif ($extension -eq ".fs") {
            Write-Log "Applying F# improvements..." "Green"
            
            # Add open System.Linq if not present
            if (-not ($improvedContent -match "open System\.Linq")) {
                $improvedContent = $improvedContent -replace "open System", "open System`r`nopen System.Linq"
                $improvements++
                Write-Log "  - Added open System.Linq" "Green"
            }
            
            # Replace imperative loops with functional approaches
            $oldContent = $improvedContent
            $improvedContent = $improvedContent -replace "for i in 0 .. (.*?)\.Length - 1 do", "Array.iteri (fun i ->"
            if ($oldContent -ne $improvedContent) {
                $improvements++
                Write-Log "  - Replaced imperative loops with functional approaches" "Green"
            }
        }
        else {
            Write-Log "Unsupported file type: $extension" "Yellow"
            return $false
        }
        
        # Check if any improvements were made
        if ($improvements -eq 0) {
            Write-Log "No improvements needed for $FilePath" "Yellow"
            return $false
        }
        
        # Save the improved content
        if ($ApplyImmediately) {
            $improvedContent | Set-Content -Path $FilePath -Force
            Write-Log "Applied $improvements improvements to $FilePath" "Green"
            return $true
        } else {
            $improvedPath = "$FilePath.improved"
            $improvedContent | Set-Content -Path $improvedPath -Force
            Write-Log "Saved improved version to $improvedPath" "Yellow"
            return $false
        }
    }
    catch {
        Write-Log "Error improving file $FilePath: $_" "Red"
        
        # Restore from backup
        if (Test-Path $backupPath) {
            Copy-Item -Path $backupPath -Destination $FilePath -Force
            Write-Log "Restored original file from backup" "Yellow"
        }
        
        return $false
    }
    finally {
        # Clean up backup
        if (Test-Path $backupPath) {
            Remove-Item -Path $backupPath -Force
        }
    }
}

# Initialize log file
"" | Set-Content -Path $LogFile -Force
Write-Log "TARS Retroaction started at $(Get-Date)" "Cyan"
Write-Log "Source path: $SourcePath" "Cyan"
Write-Log "File pattern: $FilePattern" "Cyan"
Write-Log "Recursive: $Recursive" "Cyan"
Write-Log "Apply immediately: $ApplyImmediately" "Cyan"

# Find files to process
$searchOption = if ($Recursive) { "AllDirectories" } else { "TopDirectoryOnly" }
$files = Get-ChildItem -Path $SourcePath -Filter $FilePattern -Recurse:$Recursive -File

Write-Log "Found $($files.Count) files to process" "Cyan"

# Process each file
$improvedFiles = 0
foreach ($file in $files) {
    $result = Improve-File -FilePath $file.FullName
    if ($result) {
        $improvedFiles++
    }
}

Write-Log "TARS Retroaction completed at $(Get-Date)" "Cyan"
Write-Log "Improved $improvedFiles out of $($files.Count) files" "Green"

# Return the number of improved files
return $improvedFiles
