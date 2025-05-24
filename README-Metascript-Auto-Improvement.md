# Metascript Auto-Improvement with Tree-of-Thought Reasoning

This document provides instructions on how to use the Metascript Auto-Improvement pipeline with Tree-of-Thought reasoning.

## Overview

The Metascript Auto-Improvement pipeline uses Tree-of-Thought reasoning to analyze code, generate improvements, and apply those improvements. This approach allows for more sophisticated reasoning about code improvements, leading to better results.

## Prerequisites

- .NET 6.0 or later
- F# compiler (included with .NET SDK)

## Getting Started

### 1. Build the Project

```bash
dotnet build
```

### 2. Run the Metascript Tree-of-Thought Pipeline

```bash
./run_metascript_tot.ps1
```

This script will:
1. Create the necessary directories
2. Register the services
3. Run the Metascript Tree-of-Thought pipeline command

### 3. Run the Auto-Improvement Pipeline

```bash
./run_auto_improvement.ps1
```

This script will:
1. Create the necessary directories
2. Run the auto-improvement pipeline command

### 4. Examine the Results

The results will be saved in the following files:
- `tot_output/`: Directory containing the output of the Metascript Tree-of-Thought pipeline
- `auto_improvement_report.md`: Report of the auto-improvement pipeline

## Using the CLI Commands

### Metascript Tree-of-Thought Command

```bash
tars metascript-tot pipeline --template <template-file> --values <values-file> --output-dir <output-dir>
```

### Auto-Improvement Command

```bash
tars metascript-auto-improve --file <file> --type <improvement-type> --output <output-file>
```

## Integration with TARS

The Metascript Auto-Improvement pipeline is integrated with the TARS auto-improvement pipeline, allowing for more sophisticated code improvements.

### Key Components

1. **MetascriptTreeOfThoughtService**: C# wrapper for the F# Tree-of-Thought implementation
2. **MetascriptTreeOfThoughtIntegration**: Integration with the TARS auto-improvement pipeline
3. **MetascriptTreeOfThoughtCommand**: CLI command for working with metascripts using Tree-of-Thought reasoning
4. **MetascriptAutoImprovementCommand**: CLI command for running the auto-improvement pipeline

### Auto-Improvement Pipeline

The auto-improvement pipeline consists of the following steps:

1. **Analysis**: Analyze the code using Tree-of-Thought reasoning
2. **Improvement Generation**: Generate improvements using Tree-of-Thought reasoning
3. **Improvement Application**: Apply the improvements using Tree-of-Thought reasoning

## Example

Here's an example of how to use the auto-improvement pipeline:

```bash
tars metascript-auto-improve --file Samples/SampleCode.cs --type performance --output auto_improvement_report.md
```

This will:
1. Analyze the `Samples/SampleCode.cs` file
2. Generate performance improvements
3. Apply the improvements
4. Save the report to `auto_improvement_report.md`

## Next Steps

1. **Extend the Implementation**: Add support for more improvement types
2. **Enhance the Reasoning**: Improve the Tree-of-Thought reasoning for better results
3. **Integrate with More Tools**: Integrate with more tools in the TARS ecosystem
4. **Add More Tests**: Add more tests to ensure the implementation works correctly
5. **Improve Documentation**: Expand the documentation with more examples and use cases
