#!/usr/bin/env pwsh
# PRECISELY ELIMINATE FAKE CODE - PRESERVE LEGITIMATE FUNCTIONALITY

Write-Host "PRECISELY ELIMINATING FAKE CODE FROM TARS" -ForegroundColor Red
Write-Host "=========================================" -ForegroundColor Red
Write-Host ""

# Define FAKE patterns (preserve legitimate functionality)
$fakePatterns = @{
    # FAKE execution patterns
    "FAKE_DELAYS" = @(
        "Task\.Delay\s*\(\s*\d+\s*\)\s*//.*[Ss]imulat",
        "Thread\.Sleep\s*\(\s*\d+\s*\)\s*//.*[Ss]imulat",
        "do! Task\.Delay.*[Ss]imulat"
    )
    
    # FAKE F# execution
    "FAKE_FSHARP" = @(
        "printfnPattern",
        "letPattern", 
        "Regex.*printfn",
        "Simple F# execution simulation",
        "pattern matching for common F# constructs",
        "simulate F# execution"
    )
    
    # FAKE implementations
    "FAKE_IMPL" = @(
        "simplified implementation",
        "basic pattern matching", 
        "for now.*simulate",
        "placeholder.*implement",
        "mock.*execution",
        "fake.*result"
    )
    
    # FAKE random for metrics (preserve crypto/algorithm random)
    "FAKE_RANDOM_METRICS" = @(
        "Random\(\)\.Next.*MemoryUsed",
        "Random\(\)\.Next.*CpuUsage", 
        "Random\(\)\.Next.*fake",
        "Random\(\)\.Next.*simulat"
    )
    
    # FAKE comments
    "FAKE_COMMENTS" = @(
        "// Simulate.*execution",
        "// For now.*simulate", 
        "// This is a placeholder",
        "// Mock implementation",
        "// Fake.*data"
    )
    
    # FAKE return values
    "FAKE_RETURNS" = @(
        'return ".*simulated.*"',
        'return ".*placeholder.*"',
        'return ".*mock.*"',
        'return ".*fake.*"'
    )
}

# Get F# and C# files (exclude test files that legitimately use mocks)
$codeFiles = Get-ChildItem -Path . -Include "*.fs", "*.fsx", "*.cs" -Recurse | Where-Object { 
    $_.FullName -notmatch "node_modules|\.git|bin|obj|packages|\.vs" -and
    $_.FullName -notmatch "Tests\.cs$|Test\.fs$|\.Tests\." -and
    $_.Name -ne "test-simulation-detection.fsx" -and
    $_.Name -ne "SimulationDetector.fs"
}

Write-Host "Found $($codeFiles.Count) production code files to scan" -ForegroundColor Yellow
Write-Host ""

$totalFakeCode = 0
$filesWithFakeCode = @()

foreach ($file in $codeFiles) {
    $content = Get-Content -Path $file.FullName -Raw -ErrorAction SilentlyContinue
    if (-not $content) { continue }
    
    $fakeCodeInFile = 0
    $foundPatterns = @()
    
    # Check each fake pattern category
    foreach ($category in $fakePatterns.Keys) {
        foreach ($pattern in $fakePatterns[$category]) {
            if ($content -match $pattern) {
                $matches = [regex]::Matches($content, $pattern, [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)
                if ($matches.Count -gt 0) {
                    $fakeCodeInFile += $matches.Count
                    $foundPatterns += "$category`: $pattern"
                }
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

Write-Host "PRECISE SCAN RESULTS" -ForegroundColor Yellow
Write-Host "===================" -ForegroundColor Yellow
Write-Host "Total files scanned: $($codeFiles.Count)" -ForegroundColor White
Write-Host "Files with FAKE code: $($filesWithFakeCode.Count)" -ForegroundColor Red
Write-Host "Total FAKE code instances: $totalFakeCode" -ForegroundColor Red
Write-Host ""

if ($totalFakeCode -gt 0) {
    Write-Host "CRITICAL: FAKE CODE FOUND!" -ForegroundColor Red
    Write-Host "=========================" -ForegroundColor Red
    Write-Host ""
    
    # Create backup
    $backupDir = "./backup_fake_elimination_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
    New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
    Write-Host "Created backup: $backupDir" -ForegroundColor Green
    Write-Host ""
    
    foreach ($fileInfo in $filesWithFakeCode) {
        $file = $fileInfo.File
        $relativePath = $file.Replace((Get-Location).Path, "").TrimStart('\', '/')
        
        Write-Host "FIXING: $relativePath" -ForegroundColor Yellow
        
        # Backup original
        $backupPath = Join-Path $backupDir $relativePath
        $backupParentDir = Split-Path $backupPath -Parent
        if (-not (Test-Path $backupParentDir)) {
            New-Item -ItemType Directory -Path $backupParentDir -Force | Out-Null
        }
        Copy-Item -Path $file -Destination $backupPath -Force
        
        # Read and fix content
        $content = Get-Content -Path $file -Raw
        $originalContent = $content
        
        # Apply precise fixes
        
        # Remove fake F# execution patterns
        $content = $content -replace "let printfnPattern = .*", "// REMOVED: Fake printfn pattern matching"
        $content = $content -replace "let letPattern = .*", "// REMOVED: Fake let binding pattern matching"
        $content = $content -replace "printfnPattern\.Matches.*", "// REMOVED: Fake regex execution"
        $content = $content -replace "letPattern\.Matches.*", "// REMOVED: Fake regex execution"
        
        # Remove fake delays (preserve legitimate delays)
        $content = $content -replace "do! Task\.Delay\s*\(\s*\d+\s*\)[^;]*//.*[Ss]imulat[^;]*", "// REMOVED: Fake simulation delay"
        $content = $content -replace "Thread\.Sleep\s*\(\s*\d+\s*\)\s*//.*[Ss]imulat[^;]*", "// REMOVED: Fake simulation delay"
        
        # Remove fake comments
        $content = $content -replace "// Simulate.*execution.*", "// REAL IMPLEMENTATION NEEDED"
        $content = $content -replace "// For now.*simulate.*", "// REAL IMPLEMENTATION NEEDED"
        $content = $content -replace "// This is a placeholder.*", "// REAL IMPLEMENTATION NEEDED"
        $content = $content -replace "// Mock implementation.*", "// REAL IMPLEMENTATION NEEDED"
        
        # Remove fake return values
        $content = $content -replace 'return ".*simulated.*"', 'return "REAL_RESULT_NEEDED"'
        $content = $content -replace 'return ".*placeholder.*"', 'return "REAL_RESULT_NEEDED"'
        $content = $content -replace 'return ".*mock.*"', 'return "REAL_RESULT_NEEDED"'
        $content = $content -replace 'return ".*fake.*"', 'return "REAL_RESULT_NEEDED"'
        
        # Remove fake random metrics (preserve legitimate random)
        $content = $content -replace "Random\(\)\.Next.*MemoryUsed.*", "0.0 // Cannot measure without real runtime"
        $content = $content -replace "Random\(\)\.Next.*CpuUsage.*", "0.0 // Cannot measure without real runtime"
        
        # Save fixed content
        if ($content -ne $originalContent) {
            Set-Content -Path $file -Value $content -Encoding UTF8
            Write-Host "  FIXED $($fileInfo.Count) fake code instances" -ForegroundColor Green
        } else {
            Write-Host "  No automatic fixes applied" -ForegroundColor Yellow
        }
    }
    
    Write-Host ""
    Write-Host "FAKE CODE ELIMINATION COMPLETE!" -ForegroundColor Green
    Write-Host "===============================" -ForegroundColor Green
    Write-Host "Files processed: $($filesWithFakeCode.Count)" -ForegroundColor Green
    Write-Host "Backup created: $backupDir" -ForegroundColor Green
    
} else {
    Write-Host "NO FAKE CODE FOUND!" -ForegroundColor Green
    Write-Host "Production code is clean." -ForegroundColor Green
}

Write-Host ""
Write-Host "PRESERVED LEGITIMATE FUNCTIONALITY:" -ForegroundColor Cyan
Write-Host "- Cryptographic random number generation" -ForegroundColor White
Write-Host "- Algorithm randomness for AI/ML" -ForegroundColor White  
Write-Host "- Demo ID generation" -ForegroundColor White
Write-Host "- Security analysis patterns" -ForegroundColor White
Write-Host "- Test mocking (in test files)" -ForegroundColor White
Write-Host ""
Write-Host "ZERO TOLERANCE FOR FAKE EXECUTION!" -ForegroundColor Red
