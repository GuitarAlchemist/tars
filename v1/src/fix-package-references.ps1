# Fix escaped dots in PackageReference Include attributes across all .fsproj files

Write-Host "🔧 Fixing escaped dots in PackageReference Include attributes..." -ForegroundColor Yellow

# Get all .fsproj files recursively
$projectFiles = Get-ChildItem -Path "." -Filter "*.fsproj" -Recurse

$totalFiles = $projectFiles.Count
$fixedFiles = 0
$totalReplacements = 0

foreach ($file in $projectFiles) {
    Write-Host "Processing: $($file.FullName)" -ForegroundColor Cyan
    
    $content = Get-Content $file.FullName -Raw
    $originalContent = $content
    
    # Use regex to fix all escaped dots in package references
    $beforeFix = $content
    # Fix single backslash before dot
    $content = $content -replace 'Include="([^"]*?)\\\.([^"]*?)"', 'Include="$1.$2"'
    # Fix triple backslash before dot
    $content = $content -replace 'Include="([^"]*?)\\\\\\\.([^"]*?)"', 'Include="$1.$2"'
    # Fix multiple patterns
    $content = $content -replace 'Include="([^"]*?)\\\.([^"]*?)\\\.([^"]*?)"', 'Include="$1.$2.$3"'
    $content = $content -replace 'Include="([^"]*?)\\\.([^"]*?)\\\.([^"]*?)\\\.([^"]*?)"', 'Include="$1.$2.$3.$4"'
    $content = $content -replace 'Include="([^"]*?)\\\.([^"]*?)\\\.([^"]*?)\\\.([^"]*?)\\\.([^"]*?)"', 'Include="$1.$2.$3.$4.$5"'

    $fileReplacements = 0
    if ($beforeFix -ne $content) {
        $fileReplacements = 1
        Write-Host "  ✅ Fixed escaped dots in package references" -ForegroundColor Green
    }
    
    # Also fix version numbers that are too high
    $content = $content -replace 'Version="10\.0\.0"', 'Version="9.0.0"'
    $content = $content -replace 'Version="8\.0\.5"', 'Version="9.0.0"'
    $content = $content -replace 'Version="8\.0\.0"', 'Version="9.0.0"'
    
    if ($content -ne $originalContent) {
        Set-Content -Path $file.FullName -Value $content -NoNewline
        $fixedFiles++
        $totalReplacements += $fileReplacements
        Write-Host "  📝 Updated file with $fileReplacements replacements" -ForegroundColor Green
    } else {
        Write-Host "  ✅ No changes needed" -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "🎉 Package reference fixing complete!" -ForegroundColor Green
Write-Host "   Files processed: $totalFiles" -ForegroundColor White
Write-Host "   Files fixed: $fixedFiles" -ForegroundColor White
Write-Host "   Total replacements: $totalReplacements" -ForegroundColor White
