Get-ChildItem -Recurse -Path src, tests -Filter *.fs | ForEach-Object {
    $content = [System.IO.File]::ReadAllText($_.FullName)
    if ($content -match 'Seed = None \}') {
        $newContent = $content -replace '(\s+)Seed = None \}', "`$1Seed = None`r`n`$1ContextWindow = None }"
        if ($newContent -ne $content) {
            [System.IO.File]::WriteAllText($_.FullName, $newContent)
            Write-Host "Fixed: $($_.FullName)"
        }
    }
}
