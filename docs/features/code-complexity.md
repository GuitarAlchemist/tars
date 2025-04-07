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

### Maintainability Index (Coming Soon)

The maintainability index is a software metric that measures how maintainable (easy to support and change) the source code is. The maintainability index is calculated as a factored formula consisting of Lines of Code, Cyclomatic Complexity, and Halstead Volume.

### Halstead Complexity (Coming Soon)

Halstead complexity measures are software metrics introduced by Maurice Howard Halstead in 1977 as part of his treatise on establishing an empirical science of software development. Halstead complexity measures are computed using the number of operators and operands in the program.

## Supported Languages

TARS currently supports complexity analysis for the following languages:

- C#
- F# (basic support)

## Using the Code Complexity Analyzer

### Command Line Interface

You can analyze code complexity using the TARS CLI:

```bash
tarscli complexity --path <path> [--language <language>] [--type <type>] [--output <format>]
```

#### Options

- `--path`, `-p`: Path to the file or directory to analyze (required)
- `--language`, `-l`: Programming language (C# or F#)
- `--type`, `-t`: Complexity type (Cyclomatic, Cognitive, Maintainability, Halstead, or All)
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

// Get the code complexity analyzer service
var codeComplexityAnalyzer = serviceProvider.GetRequiredService<ICodeComplexityAnalyzer>();

// Analyze cyclomatic complexity
var metrics = await codeComplexityAnalyzer.AnalyzeCyclomaticComplexityAsync("path/to/file.cs", "C#");

// Process the results
foreach (var metric in metrics)
{
    Console.WriteLine($"Target: {metric.Target}");
    Console.WriteLine($"Complexity: {metric.Value}");
    Console.WriteLine($"Threshold: {metric.ThresholdValue}");
    Console.WriteLine($"Is Above Threshold: {metric.IsAboveThreshold}");
}
```

## Configuration

You can configure the complexity thresholds for different types of entities:

```csharp
// Set the cyclomatic complexity threshold for methods in C#
await codeComplexityAnalyzer.SetComplexityThresholdAsync("C#", ComplexityType.Cyclomatic, "Method", 15);
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

- **Maintainability Index**: 20 (for all entity types)

### F#

- **Cyclomatic Complexity**:
  - Function: 8
  - Module: 15
  - File: 40

- **Cognitive Complexity**:
  - Function: 12
  - Module: 25
  - File: 60

- **Maintainability Index**: 20 (for all entity types)

## Future Enhancements

The following enhancements are planned for future releases:

- Support for more languages (JavaScript, TypeScript, Python, etc.)
- Implementation of cognitive complexity analysis
- Implementation of maintainability index calculation
- Implementation of Halstead complexity measures
- Integration with code review tools
- Trend analysis for complexity metrics over time
- Visualization of complexity metrics
- Automatic refactoring suggestions
