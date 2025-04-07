# Code Duplication Detection

The Code Duplication Detection system is a key component of TARS's Intelligence Progression Measurement framework. It helps identify duplicated code patterns, which can be a sign of poor code quality and maintainability issues.

## Overview

The system provides two main types of duplication detection:

1. **Token-Based Duplication Detection**: Identifies exact or near-exact duplicated code blocks by analyzing the token sequences in the code.
2. **Semantic Duplication Detection**: Identifies code blocks that perform similar functions even if they have different syntax or structure.

## Features

- **Multi-language Support**: Analyzes code in C# and F# (with extensibility for other languages)
- **Multiple Granularity Levels**: Detects duplication at method, class, file, and project levels
- **Visualization**: Generates HTML reports with interactive visualizations
- **Threshold Configuration**: Configurable thresholds for different types of duplication
- **Integration with Intelligence Measurement**: Contributes to the overall intelligence progression metrics

## Usage

### Command Line Interface

```bash
# Analyze a single file for token-based duplication
tarscli duplication-demo --path path/to/file.cs --type token

# Analyze a project for semantic duplication
tarscli duplication-demo --path path/to/project --type semantic

# Generate an HTML report
tarscli duplication-demo --path path/to/project --output html --output-path report.html
```

### Programmatic Usage

```csharp
// Get the duplication analyzer service
var duplicationAnalyzer = serviceProvider.GetRequiredService<IDuplicationAnalyzer>();

// Analyze token-based duplication
var tokenBasedMetrics = await duplicationAnalyzer.AnalyzeTokenBasedDuplicationAsync("path/to/file.cs", "C#");

// Analyze semantic duplication
var semanticMetrics = await duplicationAnalyzer.AnalyzeSemanticDuplicationAsync("path/to/file.cs", "C#");

// Analyze all duplication metrics
var allMetrics = await duplicationAnalyzer.AnalyzeAllDuplicationMetricsAsync("path/to/file.cs", "C#");

// Analyze project-level duplication
var projectMetrics = await duplicationAnalyzer.AnalyzeProjectDuplicationAsync("path/to/project");

// Generate visualization
await duplicationAnalyzer.VisualizeDuplicationAsync("path/to/project", "C#", "report.html");
```

## Duplication Metrics

The system calculates several metrics for code duplication:

- **Duplication Percentage**: The percentage of code that is duplicated
- **Duplicated Lines of Code**: The total number of duplicated lines
- **Duplicated Block Count**: The number of duplicated code blocks
- **Similarity Percentage**: For semantic duplication, the degree of similarity between code blocks

## Duplication Levels

Duplication is categorized into four levels:

- **Low**: Less than 3% duplication
- **Moderate**: 3-10% duplication
- **High**: 10-20% duplication
- **Very High**: More than 20% duplication

## Implementation Details

### Token-Based Duplication Detection

The token-based duplication detection works by:

1. Tokenizing the source code using the language-specific parser
2. Finding sequences of tokens that are repeated in the code
3. Converting token positions to line numbers
4. Calculating duplication metrics based on the duplicated sequences

### Semantic Duplication Detection

The semantic duplication detection works by:

1. Normalizing the code by removing comments, whitespace, and renaming variables
2. Analyzing the structure of the code using abstract syntax trees
3. Comparing the control flow and variable usage patterns
4. Calculating similarity scores based on multiple factors:
   - Structural similarity (using Levenshtein distance)
   - Variable usage similarity (using Jaccard similarity)
   - Control flow similarity (comparing if/for/while/switch statements)

## Integration with Intelligence Progression Measurement

The duplication metrics are integrated into the Intelligence Progression Measurement system to:

1. Track the system's ability to identify and reduce code duplication over time
2. Contribute to the overall code quality assessment
3. Provide insights into the system's understanding of code patterns and structures

## Future Enhancements

- **Cross-Language Duplication Detection**: Detect duplication across different programming languages
- **Machine Learning-Based Detection**: Use machine learning to improve semantic duplication detection
- **Refactoring Suggestions**: Automatically suggest refactorings to reduce duplication
- **Historical Trend Analysis**: Track duplication metrics over time to identify trends
