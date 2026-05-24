# Fix F# string interpolation format specifiers in TARS CLI
Write-Host "Fixing F# string interpolation format specifiers in TARS CLI..."

$files = Get-ChildItem -Path "TarsEngine.FSharp.Cli" -Filter "*.fs" -Recurse

foreach ($file in $files) {
    $content = Get-Content $file.FullName -Raw
    $originalContent = $content
    
    # Fix common format specifiers
    $content = $content -replace '\{([^}]+):F(\d+)\}', '{$1.ToString("F$2")}'
    $content = $content -replace '\{([^}]+):P(\d+)\}', '{$1.ToString("P$2")}'
    $content = $content -replace '\{([^}]+):HH:mm:ss\}', '{$1.ToString("HH:mm:ss")}'
    $content = $content -replace '\{([^}]+):yyyy-MM-dd HH:mm:ss\}', '{$1.ToString("yyyy-MM-dd HH:mm:ss")}'
    $content = $content -replace '\{([^}]+):hh\\:mm\\:ss\}', '{$1.ToString(@"hh\:mm\:ss")}'
    
    if ($content -ne $originalContent) {
        Set-Content $file.FullName $content
        Write-Host "Fixed: $($file.Name)"
    }
}

Write-Host "CLI string interpolation fixes complete!"
