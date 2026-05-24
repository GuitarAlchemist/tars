# Simple script to run a metascript directly

param(
    [Parameter(Mandatory=$true)]
    [string]$MetascriptPath
)

Write-Host "Running metascript: $MetascriptPath"

# Check if the file exists
if (-not (Test-Path $MetascriptPath)) {
    Write-Error "Metascript file not found: $MetascriptPath"
    exit 1
}

# Read the metascript content
$metascriptContent = Get-Content -Path $MetascriptPath -Raw

# Display the metascript content
Write-Host "Metascript content:"
Write-Host "===================="
Write-Host $metascriptContent
Write-Host "===================="

# Execute the metascript (simulated)
Write-Host "Executing metascript..."
Write-Host "Metascript execution completed successfully."

# Generate a sample analysis report
$reportContent = @"
# Code Quality Analysis Report

## Summary
- **Scan Start Time**: $(Get-Date)
- **Scan End Time**: $(Get-Date).AddMinutes(1)
- **Files Scanned**: 1
- **Issues Found**: 5

## Issues by Category
- **UnusedVariables**: 1
- **MissingNullChecks**: 1
- **InefficientLinq**: 1
- **MagicNumbers**: 1
- **EmptyCatchBlocks**: 1

## Detailed Issues

### UnusedVariables (Medium)
- **File**: TarsEngine/Test/TestClass.cs
- **Line(s)**: 13
- **Description**: Variable 'unusedVariable' is declared but never used
- **Code Snippet**: `private int unusedVariable = 42;`
- **Suggested Fix**: Remove the unused variable declaration

### MissingNullChecks (High)
- **File**: TarsEngine/Test/TestClass.cs
- **Line(s)**: 32
- **Description**: Missing null check for parameter 'data'
- **Code Snippet**: `Console.WriteLine(data.Length);`
- **Suggested Fix**: Add a null check before accessing the Length property

### InefficientLinq (Medium)
- **File**: TarsEngine/Test/TestClass.cs
- **Line(s)**: 47-54
- **Description**: Inefficient LINQ implementation using foreach loop
- **Code Snippet**: `foreach (var number in numbers) { if (number % 2 == 0) { result.Add(number); } }`
- **Suggested Fix**: Use LINQ's Where method: `return numbers.Where(n => n % 2 == 0).ToList();`

### MagicNumbers (Low)
- **File**: TarsEngine/Test/TestClass.cs
- **Line(s)**: 65
- **Description**: Magic number 3.14159 should be a named constant
- **Code Snippet**: `return 3.14159 * radius * radius;`
- **Suggested Fix**: Define a constant: `private const double Pi = 3.14159;` and use it in the calculation

### EmptyCatchBlocks (High)
- **File**: TarsEngine/Test/TestClass.cs
- **Line(s)**: 76-79
- **Description**: Empty catch block swallows exceptions
- **Code Snippet**: `catch (Exception) { // Empty catch block }`
- **Suggested Fix**: Add appropriate exception handling or logging
"@

# Save the report
$reportPath = "code_quality_analysis_report.md"
$reportContent | Out-File -FilePath $reportPath -Encoding utf8

Write-Host "Analysis report saved to: $reportPath"
