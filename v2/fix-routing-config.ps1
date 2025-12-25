Get-ChildItem -Recurse -Path src, tests -Filter *.fs | ForEach-Object {
    $content = [System.IO.File]::ReadAllText($_.FullName)
    if ($content -match 'LlamaSharpModelPath = None \}') {
        $newContent = $content -replace '(\s+)LlamaSharpModelPath = None \}', "`$1LlamaSharpModelPath = None`r`n`$1DefaultContextWindow = None`r`n`$1DefaultTemperature = None }"
        if ($newContent -ne $content) {
            [System.IO.File]::WriteAllText($_.FullName, $newContent)
            Write-Host "Fixed RoutingConfig: $($_.FullName)"
        }
    }
}
