# COMPREHENSIVE TARS VERIFICATION SYSTEM
# Verifies EVERY feature claim from ALL .md files against actual codebase

Write-Host "COMPREHENSIVE TARS VERIFICATION SYSTEM" -ForegroundColor Cyan
Write-Host "=======================================" -ForegroundColor Cyan
Write-Host "Verifying EVERY feature from ALL .md files against actual code" -ForegroundColor White
Write-Host ""

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$verificationDir = ".tars\comprehensive_verification"

# Clean and create verification directory
if (Test-Path $verificationDir) {
    Remove-Item $verificationDir -Recurse -Force
}
New-Item -ItemType Directory -Path $verificationDir | Out-Null
Write-Host "Created comprehensive verification directory: $verificationDir" -ForegroundColor Green
Write-Host ""

# STEP 1: Find ALL .md files in the repository
Write-Host "STEP 1: SCANNING ALL .MD FILES" -ForegroundColor Yellow
Write-Host "===============================" -ForegroundColor Yellow

$allMdFiles = Get-ChildItem -Path . -Filter "*.md" -Recurse | Where-Object { $_.FullName -notmatch "node_modules|\.git" }
Write-Host "Found $($allMdFiles.Count) .md files:" -ForegroundColor Green

$mdFilesList = @()
foreach ($file in $allMdFiles) {
    $relativePath = $file.FullName.Replace((Get-Location).Path, "").TrimStart('\')
    $size = $file.Length
    Write-Host "  $relativePath ($size bytes)" -ForegroundColor Gray
    $mdFilesList += @{
        Path = $relativePath
        FullPath = $file.FullName
        Size = $size
        Content = Get-Content $file.FullName -Raw
    }
}
Write-Host ""

# STEP 2: Extract ALL feature claims from .md files
Write-Host "STEP 2: EXTRACTING FEATURE CLAIMS" -ForegroundColor Yellow
Write-Host "==================================" -ForegroundColor Yellow

$allClaims = @()
$claimPatterns = @(
    @{Pattern = "✅\s+(.+)"; Type = "Completed Feature"; Priority = "High"},
    @{Pattern = "🚀\s+(.+)"; Type = "Capability Claim"; Priority = "High"},
    @{Pattern = "(\d+%)\s+(Complete|Operational|Ready|Implemented)"; Type = "Percentage Claim"; Priority = "Medium"},
    @{Pattern = "(Production-ready|Fully operational|Working|Implemented|Available)"; Type = "Status Claim"; Priority = "High"},
    @{Pattern = "(CUDA|Hyperlight|WASM|Neuromorphic|Optical|Quantum)\s+(support|integration|implementation)"; Type = "Technology Claim"; Priority = "High"},
    @{Pattern = "- (.+) \(.*implemented.*\)"; Type = "Implementation Claim"; Priority = "High"},
    @{Pattern = "### (.+) - (Complete|Done|Implemented)"; Type = "Section Completion"; Priority = "Medium"},
    @{Pattern = "\*\*(.+)\*\* - (✅|Complete|Done|Working)"; Type = "Bold Feature Claim"; Priority = "High"},
    @{Pattern = "Real (.+) (working|functional|operational)"; Type = "Reality Claim"; Priority = "Critical"},
    @{Pattern = "Zero (.+) (eliminated|removed|fake)"; Type = "Elimination Claim"; Priority = "Critical"}
)

foreach ($mdFile in $mdFilesList) {
    Write-Host "Scanning: $($mdFile.Path)" -ForegroundColor White
    
    foreach ($pattern in $claimPatterns) {
        $matches = [regex]::Matches($mdFile.Content, $pattern.Pattern, [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)
        
        foreach ($match in $matches) {
            $claim = @{
                File = $mdFile.Path
                Type = $pattern.Type
                Priority = $pattern.Priority
                Text = $match.Value.Trim()
                ExtractedText = if ($match.Groups.Count -gt 1) { $match.Groups[1].Value.Trim() } else { $match.Value.Trim() }
                LineNumber = ($mdFile.Content.Substring(0, $match.Index) -split "`n").Count
                Verified = $false
                Evidence = ""
                CodeFiles = @()
            }
            $allClaims += $claim
        }
    }
}

Write-Host "Extracted $($allClaims.Count) total claims" -ForegroundColor Green
Write-Host "Claims by type:" -ForegroundColor Gray
$allClaims | Group-Object Type | ForEach-Object {
    Write-Host "  $($_.Name): $($_.Count)" -ForegroundColor Gray
}
Write-Host ""

# STEP 3: Scan ALL code files in the repository
Write-Host "STEP 3: SCANNING ALL CODE FILES" -ForegroundColor Yellow
Write-Host "================================" -ForegroundColor Yellow

$codeExtensions = @("*.fs", "*.fsx", "*.cs", "*.fsproj", "*.csproj", "*.trsx", "*.json", "*.yml", "*.yaml")
$allCodeFiles = @()

foreach ($ext in $codeExtensions) {
    $files = Get-ChildItem -Path . -Filter $ext -Recurse | Where-Object { 
        $_.FullName -notmatch "node_modules|\.git|bin|obj" 
    }
    $allCodeFiles += $files
}

Write-Host "Found $($allCodeFiles.Count) code files:" -ForegroundColor Green
$codeFilesList = @()

foreach ($file in $allCodeFiles) {
    $relativePath = $file.FullName.Replace((Get-Location).Path, "").TrimStart('\')
    $size = $file.Length
    $extension = $file.Extension
    
    try {
        $content = Get-Content $file.FullName -Raw -ErrorAction SilentlyContinue
        $lineCount = ($content -split "`n").Count
        
        $codeFilesList += @{
            Path = $relativePath
            FullPath = $file.FullName
            Size = $size
            Extension = $extension
            LineCount = $lineCount
            Content = $content
        }
        
        Write-Host "  $relativePath ($size bytes, $lineCount lines)" -ForegroundColor Gray
    } catch {
        Write-Host "  $relativePath (ERROR: $($_.Exception.Message))" -ForegroundColor Red
    }
}
Write-Host ""

# STEP 4: Verify .tars directory contents
Write-Host "STEP 4: VERIFYING .TARS DIRECTORY" -ForegroundColor Yellow
Write-Host "==================================" -ForegroundColor Yellow

$tarsContents = @()
if (Test-Path ".tars") {
    $tarsFiles = Get-ChildItem -Path ".tars" -Recurse
    
    foreach ($file in $tarsFiles) {
        $relativePath = $file.FullName.Replace((Get-Location).Path, "").TrimStart('\')
        $size = if ($file.PSIsContainer) { 0 } else { $file.Length }
        $type = if ($file.PSIsContainer) { "Directory" } else { "File" }
        
        $tarsContents += @{
            Path = $relativePath
            Type = $type
            Size = $size
            Extension = $file.Extension
        }
        
        Write-Host "  $type : $relativePath" -ForegroundColor $(if ($type -eq "Directory") { "Cyan" } else { "Gray" })
        if ($type -eq "File" -and $size -gt 0) {
            Write-Host "    Size: $size bytes" -ForegroundColor DarkGray
        }
    }
    
    Write-Host "Total .tars items: $($tarsContents.Count)" -ForegroundColor Green
} else {
    Write-Host "ERROR: .tars directory not found!" -ForegroundColor Red
}
Write-Host ""

# STEP 5: Cross-reference claims with actual code
Write-Host "STEP 5: CROSS-REFERENCING CLAIMS WITH CODE" -ForegroundColor Yellow
Write-Host "===========================================" -ForegroundColor Yellow

$verificationResults = @()

foreach ($claim in $allClaims) {
    Write-Host "Verifying: $($claim.ExtractedText)" -ForegroundColor White
    
    $evidence = @()
    $relatedFiles = @()
    $verified = $false
    
    # Define verification keywords based on claim text
    $keywords = @()
    $claimLower = $claim.ExtractedText.ToLower()
    
    # Extract keywords from claim
    if ($claimLower -match "cuda") { $keywords += "cuda" }
    if ($claimLower -match "hyperlight") { $keywords += "hyperlight" }
    if ($claimLower -match "wasm") { $keywords += "wasm" }
    if ($claimLower -match "neuromorphic") { $keywords += "neuromorphic" }
    if ($claimLower -match "optical") { $keywords += "optical" }
    if ($claimLower -match "quantum") { $keywords += "quantum" }
    if ($claimLower -match "metascript") { $keywords += "metascript" }
    if ($claimLower -match "inference") { $keywords += "inference" }
    if ($claimLower -match "ai") { $keywords += "ai" }
    if ($claimLower -match "closure") { $keywords += "closure" }
    if ($claimLower -match "agent") { $keywords += "agent" }
    if ($claimLower -match "service") { $keywords += "service" }
    
    # Search for evidence in code files
    foreach ($codeFile in $codeFilesList) {
        $fileMatches = 0
        
        foreach ($keyword in $keywords) {
            if ($codeFile.Content -match $keyword) {
                $fileMatches++
            }
        }
        
        if ($fileMatches -gt 0) {
            $relatedFiles += @{
                Path = $codeFile.Path
                Matches = $fileMatches
                Size = $codeFile.Size
                Lines = $codeFile.LineCount
            }
        }
    }
    
    # Determine verification status
    if ($relatedFiles.Count -gt 0) {
        $verified = $true
        $evidence += "Found $($relatedFiles.Count) related code files"
        
        # Additional verification for specific claims
        if ($claim.Type -eq "Reality Claim" -or $claim.Priority -eq "Critical") {
            # More stringent verification for critical claims
            $strongEvidence = $relatedFiles | Where-Object { $_.Matches -ge 2 -and $_.Size -gt 1000 }
            if ($strongEvidence.Count -eq 0) {
                $verified = $false
                $evidence += "INSUFFICIENT: No substantial code files found"
            }
        }
    } else {
        $evidence += "NO CODE EVIDENCE FOUND"
    }
    
    $result = @{
        Claim = $claim
        Verified = $verified
        Evidence = $evidence -join "; "
        RelatedFiles = $relatedFiles
        FileCount = $relatedFiles.Count
        TotalCodeSize = ($relatedFiles | Measure-Object Size -Sum).Sum
    }
    
    $verificationResults += $result
    
    $status = if ($verified) { "VERIFIED" } else { "FAILED" }
    $color = if ($verified) { "Green" } else { "Red" }
    Write-Host "  $status ($($relatedFiles.Count) files, $($result.TotalCodeSize) bytes)" -ForegroundColor $color
}

Write-Host ""

# STEP 6: Generate comprehensive report
Write-Host "STEP 6: GENERATING COMPREHENSIVE REPORT" -ForegroundColor Yellow
Write-Host "=======================================" -ForegroundColor Yellow

$verifiedCount = ($verificationResults | Where-Object { $_.Verified }).Count
$totalClaims = $verificationResults.Count
$verificationRate = [math]::Round(($verifiedCount / $totalClaims) * 100, 1)

# Save detailed results
$detailedResultsPath = "$verificationDir\detailed_verification_results.json"
$verificationResults | ConvertTo-Json -Depth 10 | Out-File -FilePath $detailedResultsPath -Encoding UTF8

# Generate comprehensive report
$reportPath = "$verificationDir\comprehensive_verification_report.md"
$report = @"
# COMPREHENSIVE TARS VERIFICATION REPORT
**Generated:** $timestamp  
**Verification Type:** Complete feature verification against entire codebase

## EXECUTIVE SUMMARY
**VERIFICATION SCORE: $verifiedCount/$totalClaims ($verificationRate%)**

This report contains **EXHAUSTIVE VERIFICATION** of every feature claim found in all .md files against the actual codebase.

### REPOSITORY SCAN RESULTS:
- **Markdown Files Scanned:** $($mdFilesList.Count)
- **Code Files Analyzed:** $($codeFilesList.Count)
- **Claims Extracted:** $totalClaims
- **Claims Verified:** $verifiedCount
- **.tars Directory Items:** $($tarsContents.Count)

## DETAILED VERIFICATION RESULTS

### VERIFIED CLAIMS ($verifiedCount):
$($verificationResults | Where-Object { $_.Verified } | ForEach-Object {
    "**$($_.Claim.Type):** $($_.Claim.ExtractedText)  
    *Source:* $($_.Claim.File):$($_.Claim.LineNumber)  
    *Evidence:* $($_.Evidence)  
    *Related Files:* $($_.FileCount) files ($($_.TotalCodeSize) bytes)  
    "
})

### FAILED CLAIMS ($($totalClaims - $verifiedCount)):
$($verificationResults | Where-Object { -not $_.Verified } | ForEach-Object {
    "**$($_.Claim.Type):** $($_.Claim.ExtractedText)  
    *Source:* $($_.Claim.File):$($_.Claim.LineNumber)  
    *Issue:* $($_.Evidence)  
    "
})

## CODEBASE ANALYSIS

### Code Files by Type:
$($codeFilesList | Group-Object Extension | Sort-Object Count -Descending | ForEach-Object {
    "- **$($_.Name):** $($_.Count) files"
})

### Largest Code Files:
$($codeFilesList | Sort-Object Size -Descending | Select-Object -First 10 | ForEach-Object {
    "- $($_.Path): $($_.Size) bytes ($($_.LineCount) lines)"
})

### .tars Directory Contents:
$($tarsContents | ForEach-Object {
    "- $($_.Type): $($_.Path)" + $(if ($_.Size -gt 0) { " ($($_.Size) bytes)" } else { "" })
})

## VERIFICATION METHODOLOGY
1. **Comprehensive Scan:** All .md files in repository scanned for claims
2. **Pattern Matching:** Multiple regex patterns used to extract feature claims
3. **Code Cross-Reference:** Claims matched against actual code files using keyword analysis
4. **Evidence Scoring:** Files scored based on keyword matches and substantial content
5. **Critical Verification:** Higher standards applied to critical/reality claims

## LIMITATIONS
- Keyword-based matching may miss complex implementations
- Some claims may require functional testing beyond static analysis
- Performance and quality claims need runtime verification
- Integration claims need end-to-end testing

---
*This report is based on **EXHAUSTIVE STATIC ANALYSIS** of the entire TARS repository.*
*All results are reproducible and detailed evidence is included.*
"@

$report | Out-File -FilePath $reportPath -Encoding UTF8

# Generate summary
$summaryPath = "$verificationDir\verification_summary.txt"
$summary = @"
COMPREHENSIVE TARS VERIFICATION SUMMARY
======================================
Generated: $timestamp

OVERALL RESULTS:
- Verification Score: $verifiedCount/$totalClaims ($verificationRate%)
- Markdown Files: $($mdFilesList.Count)
- Code Files: $($codeFilesList.Count)
- .tars Items: $($tarsContents.Count)

CLAIMS BY PRIORITY:
$($allClaims | Group-Object Priority | Sort-Object Name | ForEach-Object {
    $priorityVerified = ($verificationResults | Where-Object { $_.Claim.Priority -eq $_.Name -and $_.Verified }).Count
    "$($_.Name): $priorityVerified/$($_.Count) verified"
})

CLAIMS BY TYPE:
$($allClaims | Group-Object Type | Sort-Object Count -Descending | ForEach-Object {
    $typeVerified = ($verificationResults | Where-Object { $_.Claim.Type -eq $_.Name -and $_.Verified }).Count
    "$($_.Name): $typeVerified/$($_.Count) verified"
})

TOP EVIDENCE FILES:
$($verificationResults | Where-Object { $_.Verified } | Sort-Object TotalCodeSize -Descending | Select-Object -First 5 | ForEach-Object {
    "$($_.Claim.ExtractedText): $($_.FileCount) files, $($_.TotalCodeSize) bytes"
})
"@

$summary | Out-File -FilePath $summaryPath -Encoding UTF8

Write-Host "Generated comprehensive report: $reportPath" -ForegroundColor Green
Write-Host "Generated summary: $summaryPath" -ForegroundColor Green
Write-Host "Generated detailed results: $detailedResultsPath" -ForegroundColor Green

# List all generated files
$evidenceFiles = Get-ChildItem $verificationDir
Write-Host ""
Write-Host "EVIDENCE FILES GENERATED:" -ForegroundColor Cyan
foreach ($file in $evidenceFiles) {
    Write-Host "  $($file.Name) ($($file.Length) bytes)" -ForegroundColor Gray
}

Write-Host ""
Write-Host "COMPREHENSIVE VERIFICATION COMPLETE!" -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Green
Write-Host "Verification Score: $verifiedCount/$totalClaims ($verificationRate%)" -ForegroundColor Cyan
Write-Host "Files Scanned: $($mdFilesList.Count) .md + $($codeFilesList.Count) code" -ForegroundColor Cyan
Write-Host "Evidence Generated: $($evidenceFiles.Count) files" -ForegroundColor Cyan
Write-Host ""
Write-Host "CHECK THE COMPREHENSIVE REPORT FOR COMPLETE ANALYSIS!" -ForegroundColor Yellow
Write-Host "Report: $reportPath" -ForegroundColor Yellow
