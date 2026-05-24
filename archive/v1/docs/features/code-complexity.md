# Code Complexity Analysis

TARS provides tools for analyzing code complexity, which is an important aspect of code quality and maintainability. The code complexity analyzer can help identify areas of code that may be difficult to understand, test, or maintain.

## Complexity Metrics

TARS currently supports the following complexity metrics:

### Cyclomatic Complexity

Cyclomatic complexity is a software metric used to indicate the complexity of a program. It is a quantitative measure of the number of linearly independent paths through a program's source code. It was developed by Thomas J. McCabe in 1976.

The cyclomatic complexity of a section of source code is the count of the number of linearly independent paths through the source code. For instance, if the source code contains no decision points such as IF statements or FOR loops, its cyclomatic complexity will be 1, since there is only a single path through the code. If the code has one single IF statement containing a simple condition, then there are two paths through the code: one where the IF statement evaluates to TRUE and another one where it evaluates to FALSE, so the cyclomatic complexity is 2.

Generally, a cyclomatic complexity of 1-10 is considered simple and easy to maintain. A complexity of 11-20 is considered moderate complexity, and anything above 20 is considered complex and difficult to maintain.

### Cognitive Complexity (Coming Soon)

Cognitive complexity is a measure of how difficult it is to understand a piece of code. Unlike cyclomatic complexity, which focuses on the number of paths through the code, cognitive complexity focuses on how difficult it is for a human to understand the code.

### Maintainability Index

The maintainability index is a software metric that measures how maintainable (easy to support and change) the source code is. The maintainability index is calculated as a factored formula consisting of Lines of Code, Cyclomatic Complexity, and Halstead Volume.

TARS supports two versions of the maintainability index formula:

1. **Original Formula**: MI = 171 - 5.2 * ln(V) - 0.23 * G - 16.2 * ln(LOC) + 50 * sin(sqrt(2.4 * CM))
   - V = Halstead Volume
   - G = Cyclomatic Complexity
   - LOC = Lines of Code
   - CM = Comment Percentage (0-1)

2. **Microsoft Formula**: MI = MAX(0, (171 - 5.2 * ln(V) - 0.23 * G - 16.2 * ln(LOC)) * 100 / 171)
   - V = Halstead Volume
   - G = Cyclomatic Complexity
   - LOC = Lines of Code

The maintainability index ranges from 0 to 100, with higher values indicating better maintainability. TARS classifies maintainability into four levels:

- **High**: MI >= 80
- **Moderate**: 60 <= MI < 80
- **Low**: 40 <= MI < 60
- **Very Low**: MI < 40

### Halstead Complexity

Halstead complexity measures are software metrics introduced by Maurice Howard Halstead in 1977 as part of his treatise on establishing an empirical science of software development. Halstead complexity measures are computed using the number of operators and operands in the program.

TARS calculates the following Halstead metrics:

1. **Program Vocabulary (n)**: The sum of the number of unique operators (n1) and unique operands (n2).
   - n = n1 + n2

2. **Program Length (N)**: The sum of the total number of operators (N1) and total number of operands (N2).
   - N = N1 + N2

3. **Program Volume (V)**: The size of the implementation of an algorithm.
   - V = N * log2(n)

4. **Program Difficulty (D)**: The difficulty of the program to write or understand.
   - D = (n1/2) * (N2/n2)

5. **Program Effort (E)**: The effort required to implement or understand a program.
   - E = D * V

6. **Time Required to Program (T)**: The estimated time to implement or understand a program.
   - T = E / 18 (in seconds)

7. **Number of Delivered Bugs (B)**: The estimated number of errors in the implementation.
   - B = E^(2/3) / 3000

## Supported Languages

TARS currently supports complexity analysis for the following languages:

- C#
- F# (basic support)

## Using the Code Complexity Analyzer

### Command Line Interface

You can analyze code complexity using the TARS CLI:

```bash
tarscli complexity --path <path> [--language <language>] [--type <type>] [--halstead-type <halstead-type>] [--output <format>]
```

#### Options

- `--path`, `-p`: Path to the file or directory to analyze (required)
- `--language`, `-l`: Programming language (C# or F#)
- `--type`, `-t`: Complexity type (Cyclomatic, Cognitive, Maintainability, Halstead, or All)
- `--halstead-type`, `-h`: Halstead metric type (Vocabulary, Length, Volume, Difficulty, Effort, TimeRequired, DeliveredBugs)
- `--output`, `-o`: Output format (Console, Json, or Csv)

#### Examples

Analyze a single C# file:

```bash
tarscli complexity --path src/MyProject/Program.cs
```

Analyze all C# files in a directory:

```bash
tarscli complexity --path src/MyProject --language C#
```

Analyze cyclomatic complexity only:

```bash
tarscli complexity --path src/MyProject --type Cyclomatic
```

Output results as JSON:

```bash
tarscli complexity --path src/MyProject --output Json
```

Analyze Halstead Volume only:

```bash
tarscli complexity --path src/MyProject --type Halstead --halstead-type Volume
```

Analyze Maintainability Index:

```bash
tarscli complexity --path src/MyProject --type Maintainability
```

### Demo

You can also run a demo of the code complexity analyzer:

```bash
tarscli demo code-complexity
```

This will create sample C# files with varying complexity and analyze them, showing the results and recommendations.

## Interpreting Results

The code complexity analyzer provides the following information for each analyzed entity (method, class, file, etc.):

- **Target**: The name of the entity being analyzed
- **Complexity**: The complexity value
- **Threshold**: The recommended threshold for this type of entity
- **Status**: Whether the complexity is within acceptable limits

If the complexity exceeds the threshold, the analyzer will provide recommendations for reducing complexity, such as:

- Breaking down methods into smaller, more focused methods
- Reducing nested conditionals by extracting helper methods
- Using design patterns to simplify complex logic
- Using early returns to reduce nesting

## Programmatic Usage

You can also use the code complexity analyzer programmatically in your own code:

```csharp
using TarsEngine.Services.Interfaces;
using TarsEngine.Models.Metrics;

// Get the code complexity analyzer service
var codeComplexityAnalyzer = serviceProvider.GetRequiredService<ICodeComplexityAnalyzer>();

// Analyze cyclomatic complexity
var complexityMetrics = await codeComplexityAnalyzer.AnalyzeCyclomaticComplexityAsync("path/to/file.cs", "C#");

// Process the complexity metrics
foreach (var metric in complexityMetrics)
{
    Console.WriteLine($"Target: {metric.Target}");
    Console.WriteLine($"Complexity: {metric.Value}");
    Console.WriteLine($"Threshold: {metric.ThresholdValue}");
    Console.WriteLine($"Is Above Threshold: {metric.IsAboveThreshold}");
}

// Analyze Halstead complexity
var halsteadMetrics = await codeComplexityAnalyzer.AnalyzeHalsteadComplexityAsync("path/to/file.cs", "C#");

// Process the Halstead metrics
foreach (var metric in halsteadMetrics.Where(m => m.Type == HalsteadType.Volume))
{
    Console.WriteLine($"Target: {metric.Target}");
    Console.WriteLine($"Halstead Volume: {metric.Volume}");
    Console.WriteLine($"Threshold: {metric.ThresholdValue}");
    Console.WriteLine($"Is Above Threshold: {metric.IsAboveThreshold}");
}

// Analyze maintainability index
var maintainabilityMetrics = await codeComplexityAnalyzer.AnalyzeMaintainabilityIndexAsync("path/to/file.cs", "C#");

// Process the maintainability metrics
foreach (var metric in maintainabilityMetrics)
{
    Console.WriteLine($"Target: {metric.Target}");
    Console.WriteLine($"Maintainability Index: {metric.Value}");
    Console.WriteLine($"Maintainability Level: {metric.MaintainabilityLevel}");
    Console.WriteLine($"Threshold: {metric.ThresholdValue}");
    Console.WriteLine($"Is Below Threshold: {metric.IsBelowThreshold}");
}

// Analyze all metrics at once
var allMetrics = await codeComplexityAnalyzer.AnalyzeAllComplexityMetricsAsync("path/to/file.cs", "C#");

// Process all metrics
Console.WriteLine($"Complexity Metrics: {allMetrics.ComplexityMetrics.Count}");
Console.WriteLine($"Halstead Metrics: {allMetrics.HalsteadMetrics.Count}");
Console.WriteLine($"Maintainability Metrics: {allMetrics.MaintainabilityMetrics.Count}");
```

## Configuration

You can configure the complexity thresholds for different types of entities:

```csharp
// Set the cyclomatic complexity threshold for methods in C#
await codeComplexityAnalyzer.SetComplexityThresholdAsync("C#", ComplexityType.Cyclomatic, "Method", 15);

// Set the Halstead volume threshold for methods in C#
await codeComplexityAnalyzer.SetHalsteadThresholdAsync("C#", HalsteadType.Volume, "Method", 500);

// Set the maintainability index threshold for methods in C#
await codeComplexityAnalyzer.SetMaintainabilityThresholdAsync("C#", "Method", 60);
```

The default thresholds are:

### C#

- **Cyclomatic Complexity**:
  - Method: 10
  - Class: 20
  - File: 50

- **Cognitive Complexity**:
  - Method: 15
  - Class: 30
  - File: 75

- **Halstead Volume**:
  - Method: 500
  - Class: 4000
  - File: 10000

- **Halstead Difficulty**:
  - Method: 15
  - Class: 30
  - File: 50

- **Halstead Effort**:
  - Method: 10000
  - Class: 100000
  - File: 500000

- **Maintainability Index**:
  - Method: 60
  - Class: 50
  - File: 40

### F#

- **Cyclomatic Complexity**:
  - Function: 8
  - Module: 15
  - File: 40

- **Cognitive Complexity**:
  - Function: 12
  - Module: 25
  - File: 60

- **Halstead Volume**:
  - Function: 500
  - Module: 4000
  - File: 10000

- **Halstead Difficulty**:
  - Function: 15
  - Module: 30
  - File: 50

- **Halstead Effort**:
  - Function: 10000
  - Module: 100000
  - File: 500000

- **Maintainability Index**:
  - Function: 60
  - Module: 50
  - File: 40

## Future Enhancements

The following enhancements are planned for future releases:

- Support for more languages (JavaScript, TypeScript, Python, etc.)
- Implementation of cognitive complexity analysis
- Enhanced F# support using FSharp.Compiler.Service
- Integration with code review tools
- Trend analysis for complexity metrics over time
- Visualization of complexity metrics
- Automatic refactoring suggestions
- Correlation analysis between different metrics
- Machine learning-based complexity prediction
- Integration with CI/CD pipelines for automated complexity checks
