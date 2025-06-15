#!/usr/bin/env pwsh
# SCAN FOR ALL FAKE CODE IN TARS CODEBASE

Write-Host "SCANNING FOR FAKE CODE IN TARS CODEBASE" -ForegroundColor Red
Write-Host "=======================================" -ForegroundColor Red
Write-Host ""

# Get all F# and C# files
$codeFiles = Get-ChildItem -Path . -Include "*.fs", "*.fsx", "*.cs" -Recurse | Where-Object { 
    $_.FullName -notmatch "node_modules|\.git|bin|obj|packages|\.vs" 
}

Write-Host "Found $($codeFiles.Count) code files to scan" -ForegroundColor Yellow
Write-Host ""

$totalFakeCode = 0
$filesWithFakeCode = @()

# Define fake code patterns
$fakePatterns = @(
    "Task\.Delay\s*\(\s*\d+\s*\)",
    "Thread\.Sleep\s*\(\s*\d+\s*\)",
    "simulate",
    "simulation", 
    "simulated",
    "placeholder",
    "mock",
    "fake",
    "dummy",
    "stub",
    "pretend",
    "TODO",
    "FIXME",
    "printfnPattern",
    "letPattern",
    "Regex.*printfn",
    "Regex.*let\\s+",
    "Simple F# execution simulation",
    "simplified implementation",
    "basic pattern matching",
    "for now",
    "in a full implementation",
    "would execute",
    "would run",
    "pattern matching for common F# constructs"
)

foreach ($file in $codeFiles) {
    $content = Get-Content -Path $file.FullName -Raw -ErrorAction SilentlyContinue
    if (-not $content) { continue }
    
    $fakeCodeInFile = 0
    $foundPatterns = @()
    
    foreach ($pattern in $fakePatterns) {
        if ($content -match $pattern) {
            $matches = [regex]::Matches($content, $pattern, [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)
            if ($matches.Count -gt 0) {
                $fakeCodeInFile += $matches.Count
                $foundPatterns += $pattern
            }
        }
    }
    
    if ($fakeCodeInFile -gt 0) {
        $totalFakeCode += $fakeCodeInFile
        $filesWithFakeCode += @{
            File = $file.FullName
            Count = $fakeCodeInFile
            Patterns = $foundPatterns
        }
        
        Write-Host "FAKE CODE DETECTED: $($file.Name)" -ForegroundColor Red
        Write-Host "  Instances: $fakeCodeInFile" -ForegroundColor Red
        foreach ($pattern in $foundPatterns) {
            Write-Host "  - $pattern" -ForegroundColor DarkRed
        }
        Write-Host ""
    }
}

Write-Host "SCAN RESULTS" -ForegroundColor Yellow
Write-Host "============" -ForegroundColor Yellow
Write-Host "Total files scanned: $($codeFiles.Count)" -ForegroundColor White
Write-Host "Files with fake code: $($filesWithFakeCode.Count)" -ForegroundColor Red
Write-Host "Total fake code instances: $totalFakeCode" -ForegroundColor Red
Write-Host ""

if ($totalFakeCode -gt 0) {
    Write-Host "CRITICAL: FAKE CODE FOUND IN CODEBASE!" -ForegroundColor Red
    Write-Host "ALL FAKE CODE MUST BE ELIMINATED!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Files that need fixing:" -ForegroundColor Yellow
    foreach ($fileInfo in $filesWithFakeCode) {
        $relativePath = $fileInfo.File.Replace((Get-Location).Path, "").TrimStart('\', '/')
        Write-Host "- $relativePath ($($fileInfo.Count) instances)" -ForegroundColor Red
    }
} else {
    Write-Host "NO FAKE CODE FOUND!" -ForegroundColor Green
    Write-Host "Codebase is clean." -ForegroundColor Green
}

Write-Host ""
Write-Host "ZERO TOLERANCE FOR FAKE CODE!" -ForegroundColor Red
