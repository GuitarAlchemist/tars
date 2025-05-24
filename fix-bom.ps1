# Script to remove BOM characters from F# files

# Get all F# files in the repository
$fsFiles = Get-ChildItem -Path . -Filter *.fs -Recurse

# Counter for files processed
$fixedFiles = 0

foreach ($file in $fsFiles) {
    # Read the file content as bytes
    $bytes = [System.IO.File]::ReadAllBytes($file.FullName)
    
    # Check if the file has a BOM (UTF-8 BOM is EF BB BF)
    if ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF) {
        Write-Host "Fixing BOM in file: $($file.FullName)"
        
        # Read the content as string (this will include the BOM)
        $content = [System.IO.File]::ReadAllText($file.FullName)
        
        # Create a new file without BOM
        [System.IO.File]::WriteAllText($file.FullName, $content, [System.Text.UTF8Encoding]::new($false))
        
        $fixedFiles++
    }
}

Write-Host "Fixed $fixedFiles files with BOM characters."
