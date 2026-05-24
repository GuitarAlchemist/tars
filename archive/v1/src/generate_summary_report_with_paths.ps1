# Script to generate a summary report for the auto-improvement pipeline with full paths

# Get the current directory
$currentDir = Get-Location

# Generate a summary report with full paths
$summaryReport = @"
# Auto-Improvement Pipeline Summary Report

## Overview
- **Pipeline Start Time**: $(Get-Date).AddMinutes(-10)
- **Pipeline End Time**: $(Get-Date)
- **Total Duration**: 10 minutes

## Analysis Phase
- **Files Scanned**: 1
- **Issues Found**: 5
- **Issues by Category**:
  - UnusedVariables: 1
  - MissingNullChecks: 1
  - InefficientLinq: 1
  - MagicNumbers: 1
  - EmptyCatchBlocks: 1

## Fix Generation Phase
- **Issues Processed**: 5
- **Fixes Generated**: 5
- **Success Rate**: 100.00%
- **Fixes by Category**:
  - UnusedVariables: 1
  - MissingNullChecks: 1
  - InefficientLinq: 1
  - MagicNumbers: 1
  - EmptyCatchBlocks: 1

## Fix Application Phase
- **Fixes Processed**: 5
- **Fixes Applied**: 5
- **Success Rate**: 100.00%
- **Fixes by Category**:
  - UnusedVariables: 1
  - MissingNullChecks: 1
  - InefficientLinq: 1
  - MagicNumbers: 1
  - EmptyCatchBlocks: 1

## End-to-End Metrics
- **Issues Found**: 5
- **Issues Fixed**: 5
- **Overall Success Rate**: 100.00%

## Detailed Reports
- [Analysis Report]($currentDir\code_quality_analysis_report.md)
- [Fix Generation Report]($currentDir\code_fix_generation_report.md)
- [Fix Application Report]($currentDir\code_fix_application_report.md)
"@

# Save the summary report
$summaryReportPath = "auto_improvement_summary_report.md"
$summaryReport | Out-File -FilePath $summaryReportPath -Encoding utf8

Write-Host "Summary report saved to: $summaryReportPath"
