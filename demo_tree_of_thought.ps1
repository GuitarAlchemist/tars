# Script to demonstrate the Tree-of-Thought concept

Write-Host "Demonstrating Tree-of-Thought reasoning for code improvement..."

# Create the Samples directory if it doesn't exist
if (-not (Test-Path -Path "Samples")) {
    New-Item -Path "Samples" -ItemType Directory | Out-Null
}

# Create a sample code file if it doesn't exist
$sampleCodePath = "Samples/SampleCode.cs"
if (-not (Test-Path -Path $sampleCodePath)) {
    $sampleCode = @"
using System;
using System.Collections.Generic;
using System.Linq;

namespace Samples
{
    public class SampleCode
    {
        // This method has performance issues that could be improved
        public void ProcessData(List<int> data)
        {
            // Inefficient: Creates a new list for each iteration
            for (int i = 0; i < data.Count; i++)
            {
                var filteredData = data.Where(x => x > i).ToList();
                
                // Inefficient: Uses LINQ inside a loop
                foreach (var item in filteredData)
                {
                    Console.WriteLine(`$"Processing item: {item}");
                    
                    // Inefficient: Repeated string concatenation
                    string result = "";
                    for (int j = 0; j < item; j++)
                    {
                        result += j.ToString();
                    }
                    
                    Console.WriteLine(result);
                }
            }
        }
        
        // This method has error handling issues
        public int CalculateAverage(List<int> numbers)
        {
            // Missing null check
            // Missing empty list check
            
            int sum = 0;
            foreach (var number in numbers)
            {
                sum += number;
            }
            
            // Potential division by zero
            return sum / numbers.Count;
        }
        
        // This method has maintainability issues
        public void ProcessOrder(Order order)
        {
            // Magic numbers
            if (order.Status == 1)
            {
                // Hardcoded values
                order.Total = order.Subtotal * 1.08m;
                
                // Unclear logic
                if (order.Total > 100)
                {
                    order.ApplyDiscount(5);
                }
                else if (order.Total > 50)
                {
                    order.ApplyDiscount(2);
                }
                
                // Duplicate code
                Console.WriteLine(`$"Order processed: {order.Id}");
                Console.WriteLine(`$"Order total: {order.Total}");
                Console.WriteLine(`$"Order date: {order.Date}");
            }
        }
    }
    
    public class Order
    {
        public int Id { get; set; }
        public decimal Subtotal { get; set; }
        public decimal Total { get; set; }
        public int Status { get; set; }
        public DateTime Date { get; set; }
        
        public void ApplyDiscount(decimal percentage)
        {
            Total -= Total * (percentage / 100);
        }
    }
}
"@
    $sampleCode = $sampleCode.Replace('`', '$')
    Set-Content -Path $sampleCodePath -Value $sampleCode
}

# Simulate Tree-of-Thought reasoning for code analysis
Write-Host "`nStep 1: Analyzing code using Tree-of-Thought reasoning..."
$analysisReport = @"
# Code Analysis Report

## Overview

Tree-of-Thought reasoning was used to analyze the code.

## Approaches

1. **Static Analysis** (Score: 0.8)
   - Analyzed code structure
   - Identified potential issues

2. **Pattern Matching** (Score: 0.7)
   - Matched code against known patterns
   - Identified common anti-patterns

3. **Semantic Analysis** (Score: 0.9)
   - Analyzed code semantics
   - Identified logical issues

## Selected Approach

Semantic Analysis was selected as the best approach with a score of 0.9.

## Results

The code analysis identified the following issues:
- Performance issues in ProcessData method
- Error handling issues in CalculateAverage method
- Maintainability issues in ProcessOrder method
"@
Set-Content -Path "analysis_report.md" -Value $analysisReport
Write-Host "Analysis report saved to analysis_report.md"

# Simulate Tree-of-Thought reasoning for improvement generation
Write-Host "`nStep 2: Generating improvements using Tree-of-Thought reasoning..."
$improvementsReport = @"
# Improvement Generation Report

## Overview

Tree-of-Thought reasoning was used to generate improvements for: Improve performance in Samples/SampleCode.cs

## Approaches

1. **Direct Fix** (Score: 0.7)
   - Simple, targeted fix
   - Addresses the immediate issue

2. **Refactoring** (Score: 0.9)
   - Comprehensive solution
   - Improves overall code quality

3. **Alternative Implementation** (Score: 0.6)
   - Different approach
   - May require significant changes

## Selected Approach

Refactoring was selected as the best approach with a score of 0.9.

## Suggested Improvements

1. Replace repeated string concatenation with StringBuilder
2. Move LINQ operations outside of loops
3. Use more efficient data structures
4. Add proper error handling
5. Replace magic numbers with constants
6. Extract duplicate code into methods
"@
Set-Content -Path "improvements_report.md" -Value $improvementsReport
Write-Host "Improvements report saved to improvements_report.md"

# Simulate Tree-of-Thought reasoning for improvement application
Write-Host "`nStep 3: Applying improvements using Tree-of-Thought reasoning..."
$applicationReport = @"
# Improvement Application Report

## Overview

Tree-of-Thought reasoning was used to apply improvements for: Apply performance improvements to Samples/SampleCode.cs

## Approaches

1. **In-Place Modification** (Score: 0.8)
   - Direct modification of the code
   - Minimal disruption

2. **Staged Application** (Score: 0.7)
   - Apply changes in stages
   - Easier to verify

3. **Transactional Application** (Score: 0.9)
   - All-or-nothing approach
   - Ensures consistency

## Selected Approach

Transactional Application was selected as the best approach with a score of 0.9.

## Application Results

The improvements were applied successfully:
- Replaced string concatenation with StringBuilder
- Moved LINQ operations outside of loops
- Added proper error handling
- Replaced magic numbers with constants
- Extracted duplicate code into methods
"@
Set-Content -Path "application_report.md" -Value $applicationReport
Write-Host "Application report saved to application_report.md"

# Create a summary report
Write-Host "`nCreating summary report..."
$summaryReport = @"
# Auto-Improvement Pipeline Report

## Overview

- **Target File**: Samples/SampleCode.cs
- **Improvement Type**: performance

## Pipeline Steps

### 1. Analysis

$analysisReport

### 2. Improvement Generation

$improvementsReport

### 3. Improvement Application

$applicationReport

## Conclusion

The auto-improvement pipeline completed successfully.
"@
Set-Content -Path "summary_report.md" -Value $summaryReport
Write-Host "Summary report saved to summary_report.md"

Write-Host "`nTree-of-Thought demonstration completed successfully!"
