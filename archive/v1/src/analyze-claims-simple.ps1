# TARS Documentation Claim Analysis Script
# Identifies false or unverifiable claims before TARS-L implementation

Write-Host "TARS DOCUMENTATION CLAIM ANALYSIS" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host "Objective: Clean up false/unverifiable claims before TARS-L implementation" -ForegroundColor Yellow
Write-Host ""

# Define problematic patterns
$problematicPatterns = @{
    "EXAGGERATED_CAPABILITY" = @("revolutionary", "unprecedented", "world-class", "best-in-class", "cutting-edge", "state-of-the-art")
    "UNVERIFIED_PERFORMANCE" = @("fastest", "most efficient", "optimal", "superior", "outperforms")
    "INCOMPLETE_FEATURES" = @("coming soon", "will be", "planned", "future", "TODO", "placeholder")
    "ABSOLUTE_CLAIMS" = @("always", "never", "all", "every", "complete", "perfect", "flawless")
    "VAGUE_BENEFITS" = @("significantly", "dramatically", "exponentially", "massive", "huge")
    "UNPROVEN_AI" = @("autonomous", "intelligent", "learns", "thinks", "understands", "conscious")
}

# Get all documentation files
$docFiles = Get-ChildItem -Path "." -Recurse -Include "*.md", "*.txt", "*.rst" | 
    Where-Object { $_.FullName -notmatch "(bin|obj|node_modules|\.git|packages)" }

Write-Host "Found $($docFiles.Count) documentation files to analyze" -ForegroundColor Green
Write-Host ""

$totalIssues = 0
$filesWithIssues = @()

# Analyze each file
foreach ($file in $docFiles) {
    try {
        $content = Get-Content $file.FullName -Raw -ErrorAction SilentlyContinue
        if (-not $content) { continue }
        
        $fileIssues = @()
        $relativePath = $file.FullName.Replace((Get-Location).Path, "").TrimStart('\')
        
        foreach ($category in $problematicPatterns.Keys) {
            foreach ($pattern in $problematicPatterns[$category]) {
                $matches = [regex]::Matches($content, $pattern, [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)
                foreach ($match in $matches) {
                    $lineNumber = ($content.Substring(0, $match.Index) -split "`n").Count
                    $fileIssues += @{
                        Category = $category
                        Text = $match.Value
                        Line = $lineNumber
                    }
                    $totalIssues++
                }
            }
        }
        
        if ($fileIssues.Count -gt 0) {
            $filesWithIssues += @{
                File = $relativePath
                Issues = $fileIssues
            }
        }
    }
    catch {
        Write-Warning "Error analyzing $($file.FullName): $($_.Exception.Message)"
    }
}

# Generate report
Write-Host "CLEANUP RECOMMENDATIONS:" -ForegroundColor Cyan
Write-Host "========================" -ForegroundColor Cyan
Write-Host "Total issues found: $totalIssues across $($filesWithIssues.Count) files" -ForegroundColor Yellow
Write-Host ""

# Group by category
$issuesByCategory = @{}
foreach ($fileData in $filesWithIssues) {
    foreach ($issue in $fileData.Issues) {
        if (-not $issuesByCategory.ContainsKey($issue.Category)) {
            $issuesByCategory[$issue.Category] = @()
        }
        $issuesByCategory[$issue.Category] += $issue
    }
}

# Show recommendations by category
foreach ($category in $issuesByCategory.Keys) {
    $count = $issuesByCategory[$category].Count
    Write-Host "$category ($count issues):" -ForegroundColor Yellow
    
    switch ($category) {
        "EXAGGERATED_CAPABILITY" {
            Write-Host "   - Replace with specific, measurable descriptions" -ForegroundColor White
            Write-Host "   - Use concrete examples instead of superlatives" -ForegroundColor White
            Write-Host "   - Focus on actual implemented features" -ForegroundColor White
        }
        "UNVERIFIED_PERFORMANCE" {
            Write-Host "   - Remove performance claims without benchmarks" -ForegroundColor White
            Write-Host "   - Replace with 'designed for' or 'optimized for'" -ForegroundColor White
            Write-Host "   - Add actual performance metrics if available" -ForegroundColor White
        }
        "INCOMPLETE_FEATURES" {
            Write-Host "   - Move to roadmap or TODO section" -ForegroundColor White
            Write-Host "   - Mark clearly as 'planned' or 'experimental'" -ForegroundColor White
            Write-Host "   - Remove from main feature descriptions" -ForegroundColor White
        }
        "ABSOLUTE_CLAIMS" {
            Write-Host "   - Add qualifiers like 'typically', 'generally', 'in most cases'" -ForegroundColor White
            Write-Host "   - Provide specific conditions or contexts" -ForegroundColor White
            Write-Host "   - Use more precise language" -ForegroundColor White
        }
        "VAGUE_BENEFITS" {
            Write-Host "   - Quantify benefits with specific metrics" -ForegroundColor White
            Write-Host "   - Provide concrete examples" -ForegroundColor White
            Write-Host "   - Use measurable improvements" -ForegroundColor White
        }
        "UNPROVEN_AI" {
            Write-Host "   - Clarify as 'automated' rather than 'intelligent'" -ForegroundColor White
            Write-Host "   - Describe specific algorithms or approaches" -ForegroundColor White
            Write-Host "   - Avoid anthropomorphic language" -ForegroundColor White
        }
    }
    Write-Host ""
}

# Show files with most issues
Write-Host "FILES REQUIRING MOST ATTENTION:" -ForegroundColor Cyan
$topFiles = $filesWithIssues | Sort-Object { $_.Issues.Count } -Descending | Select-Object -First 10
foreach ($fileData in $topFiles) {
    Write-Host "   $($fileData.File) ($($fileData.Issues.Count) issues)" -ForegroundColor Red
}
Write-Host ""

# Show specific examples
Write-Host "DETAILED EXAMPLES (Top 5 files):" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
foreach ($fileData in ($topFiles | Select-Object -First 5)) {
    Write-Host "$($fileData.File):" -ForegroundColor Yellow
    foreach ($issue in ($fileData.Issues | Select-Object -First 3)) {
        Write-Host "   Line $($issue.Line) [$($issue.Category)]: `"$($issue.Text)`"" -ForegroundColor White
    }
    if ($fileData.Issues.Count -gt 3) {
        Write-Host "   ... and $($fileData.Issues.Count - 3) more issues" -ForegroundColor Gray
    }
    Write-Host ""
}

# Summary
Write-Host "ANALYSIS SUMMARY:" -ForegroundColor Cyan
Write-Host "=================" -ForegroundColor Cyan
Write-Host "Documentation files scanned: $($docFiles.Count)" -ForegroundColor Green
Write-Host "Files with issues: $($filesWithIssues.Count)" -ForegroundColor Yellow
Write-Host "Total problematic claims: $totalIssues" -ForegroundColor Red
Write-Host "Issue categories: $($issuesByCategory.Keys.Count)" -ForegroundColor Blue
Write-Host ""

Write-Host "NEXT STEPS BEFORE TARS-L IMPLEMENTATION:" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "1. Review and clean up high-priority files" -ForegroundColor Green
Write-Host "2. Replace exaggerated claims with factual descriptions" -ForegroundColor Green
Write-Host "3. Move incomplete features to roadmap sections" -ForegroundColor Green
Write-Host "4. Add disclaimers for experimental features" -ForegroundColor Green
Write-Host "5. Verify all technical claims are accurate" -ForegroundColor Green
Write-Host "6. Update README files to reflect actual capabilities" -ForegroundColor Green
Write-Host ""
Write-Host "PRIORITY: Clean documentation before implementing TARS-L" -ForegroundColor Red
Write-Host "This ensures the revolutionary TARS-L system is built on" -ForegroundColor Yellow
Write-Host "a foundation of accurate, verifiable documentation." -ForegroundColor Yellow

# Save detailed report
$reportPath = "ChatGPT-TARS-Repo-Claim-Analysis.md"
$report = @"
# TARS Documentation Claim Analysis Report

**Generated:** $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
**Total Issues:** $totalIssues
**Files Analyzed:** $($docFiles.Count)
**Files with Issues:** $($filesWithIssues.Count)

## Summary by Category

"@

foreach ($category in $issuesByCategory.Keys) {
    $count = $issuesByCategory[$category].Count
    $report += "`n### $category ($count issues)`n`n"
}

$report += "`n## Files Requiring Attention`n`n"
foreach ($fileData in $topFiles) {
    $report += "- **$($fileData.File)** ($($fileData.Issues.Count) issues)`n"
}

$report += "`n## Recommendations`n`n"
$report += "1. Replace exaggerated claims with specific, measurable descriptions`n"
$report += "2. Remove unverified performance claims`n"
$report += "3. Move incomplete features to roadmap sections`n"
$report += "4. Add qualifiers to absolute statements`n"
$report += "5. Quantify vague benefits with metrics`n"
$report += "6. Use precise technical language for AI capabilities`n"

$report | Out-File -FilePath $reportPath -Encoding UTF8
Write-Host ""
Write-Host "Detailed report saved to: $reportPath" -ForegroundColor Green
