# TARS Auto-Improvement Pipeline

## Overview

The TARS Auto-Improvement Pipeline is a system that automatically analyzes, generates fixes for, and applies improvements to the TARS codebase. It focuses on code quality issues and best practices in C# code.

## Components

The pipeline consists of three main components:

1. **Code Quality Analyzer** (`code_quality_analyzer.tars`): Analyzes C# code for quality issues and best practice violations.
2. **Code Fix Generator** (`code_fix_generator.tars`): Generates fixes for identified code quality issues.
3. **Code Fix Applicator** (`code_fix_applicator.tars`): Applies validated fixes to the codebase.

These components can be run individually or as a complete pipeline using the `auto_improvement_pipeline.tars` metascript.

## Usage

### Command Line Interface

The auto-improvement pipeline can be run using the `auto-improve` command:

```
tars auto-improve [options]
```

Options:
- `--target <target>`: The target to improve (all, code-quality, documentation, tests). Default: all.
- `--dry-run`: Run in dry-run mode (don't apply changes). Default: false.
- `--verbose`: Enable verbose logging. Default: false.

Examples:
```
tars auto-improve
tars auto-improve --target code-quality
tars auto-improve --dry-run
```

### Running Individual Components

You can also run individual components of the pipeline:

1. **Code Quality Analyzer**:
   ```
   tars metascript run TarsCli/Metascripts/Improvements/code_quality_analyzer.tars
   ```

2. **Code Fix Generator**:
   ```
   tars metascript run TarsCli/Metascripts/Improvements/code_fix_generator.tars
   ```

3. **Code Fix Applicator**:
   ```
   tars metascript run TarsCli/Metascripts/Improvements/code_fix_applicator.tars
   ```

## Output

The pipeline generates several output files:

1. **Analysis Report** (`code_quality_analysis_report.md`): Detailed report of all identified issues.
2. **Fix Generation Report** (`code_fix_generation_report.md`): Report of all generated fixes.
3. **Fix Application Report** (`code_fix_application_report.md`): Report of all applied fixes.
4. **Summary Report** (`auto_improvement_summary_report.md`): Overall summary of the pipeline execution.

## Issue Categories

The analyzer looks for the following categories of issues:

1. **UnusedVariables**: Variables that are declared but never used.
2. **InefficientLinq**: LINQ queries that could be optimized.
3. **MissingNullChecks**: Places where null checks should be added.
4. **ImproperDisposable**: IDisposable objects not properly disposed.
5. **RedundantCode**: Code that is unnecessary or could be simplified.
6. **InconsistentNaming**: Names that don't follow C# conventions.
7. **MagicNumbers**: Hardcoded numbers that should be constants.
8. **LongMethods**: Methods that are too long and should be refactored.
9. **ComplexConditions**: Overly complex conditional expressions.
10. **EmptyCatchBlocks**: Empty catch blocks that swallow exceptions.

## Extending the Pipeline

To add support for new types of improvements:

1. Create a new analyzer metascript that identifies the issues.
2. Create a new fix generator metascript that generates fixes for the issues.
3. Create a new fix applicator metascript that applies the fixes.
4. Update the `auto_improvement_pipeline.tars` metascript to include the new components.
5. Update the `AutoImprovementCommand.cs` file to add support for the new target.

## Testing

The auto-improvement pipeline includes unit tests in the `TarsCli.Tests/Commands/AutoImprovementCommandTests.cs` file. These tests verify that the command works correctly with different targets and options.

To run the tests:

```
dotnet test TarsCli.Tests
```
