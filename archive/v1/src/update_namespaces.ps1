# Update namespaces in merged files
$files = Get-ChildItem -Path "TarsEngine.FSharp.SelfImprovement\*.fs"

foreach ($file in $files) {
    $content = Get-Content $file.FullName -Raw
    $newContent = $content -replace 'namespace TarsEngine\.SelfImprovement', 'namespace TarsEngine.FSharp.SelfImprovement'
    $newContent = $newContent -replace 'open TarsEngine\.SelfImprovement', 'open TarsEngine.FSharp.SelfImprovement'
    Set-Content -Path $file.FullName -Value $newContent -NoNewline
    Write-Host "Updated: $($file.Name)"
}

Write-Host "Namespace update complete!"

