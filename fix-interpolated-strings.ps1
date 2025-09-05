# PowerShell script to fix F# interpolated string errors (FS3373)
# This script converts $"..." interpolated strings to sprintf format

Write-Host "🔧 Fixing F# Interpolated String Errors (FS3373)" -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan

$ErrorActionPreference = "Continue"

# Function to fix interpolated strings in a file
function Fix-InterpolatedStrings {
    param([string]$FilePath)
    
    if (-not (Test-Path $FilePath)) {
        Write-Host "❌ File not found: $FilePath" -ForegroundColor Red
        return
    }
    
    Write-Host "🔍 Processing: $FilePath" -ForegroundColor Yellow
    
    $content = Get-Content $FilePath -Raw
    $originalContent = $content
    
    # Pattern to match $"..." interpolated strings with {variable} placeholders
    # This is a simplified approach - may need refinement for complex cases
    $pattern = '\$"([^"]*\{[^}]+\}[^"]*)"'
    
    $matches = [regex]::Matches($content, $pattern)
    $changeCount = 0
    
    foreach ($match in $matches) {
        $fullMatch = $match.Value
        $innerContent = $match.Groups[1].Value
        
        # Simple conversion: replace {variable} with %s and extract variables
        # This is a basic implementation - may need enhancement for complex cases
        $variables = [regex]::Matches($innerContent, '\{([^}]+)\}')
        $formatString = $innerContent
        $varList = @()
        
        foreach ($var in $variables) {
            $varName = $var.Groups[1].Value
            $varList += $varName
            # Replace {variable} with %s (simplified - should be type-aware)
            $formatString = $formatString -replace "\{$([regex]::Escape($varName))\}", "%s"
        }
        
        if ($varList.Count -gt 0) {
            $sprintfCall = "sprintf `"$formatString`" " + ($varList -join " ")
            $content = $content -replace [regex]::Escape($fullMatch), $sprintfCall
            $changeCount++
        }
    }
    
    if ($changeCount -gt 0) {
        Set-Content $FilePath -Value $content -NoNewline
        Write-Host "  ✅ Fixed $changeCount interpolated strings" -ForegroundColor Green
    } else {
        Write-Host "  ℹ️ No interpolated strings found" -ForegroundColor Gray
    }
}

# Find all F# files in the CLI project
$fsFiles = Get-ChildItem -Path "TarsEngine.FSharp.Cli" -Filter "*.fs" -Recurse

Write-Host "📁 Found $($fsFiles.Count) F# files to process" -ForegroundColor Cyan
Write-Host ""

foreach ($file in $fsFiles) {
    Fix-InterpolatedStrings -FilePath $file.FullName
}

Write-Host ""
Write-Host "🎉 Interpolated string fixing completed!" -ForegroundColor Green
Write-Host "📝 Note: This is a basic fix - manual review may be needed for complex cases" -ForegroundColor Yellow
