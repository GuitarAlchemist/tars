$files = Get-ChildItem -Path "TarsEngine" -Recurse -Include "*.cs" | Where-Object { $_.FullName -notlike "*\obj\*" -and $_.FullName -notlike "*\bin\*" }

foreach ($file in $files) {
    $content = Get-Content -Path $file.FullName -Raw
    
    # Replace 'private readonly Random _random = new Random();' with 'private readonly System.Random _random = new System.Random();'
    $newContent = $content -replace 'private readonly Random _random = new Random\(\);', 'private readonly System.Random _random = new System.Random();'
    
    # Replace other Random references
    $newContent = $newContent -replace 'new Random\(\)', 'new System.Random()'
    
    # Write the content back to the file if changes were made
    if ($content -ne $newContent) {
        Write-Host "Fixing Random namespace in $($file.FullName)"
        Set-Content -Path $file.FullName -Value $newContent
    }
}
