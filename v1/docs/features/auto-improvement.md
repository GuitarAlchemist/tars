# TARS Auto-Improvement

The TARS Auto-Improvement feature allows TARS to autonomously improve its own codebase and documentation. This document describes the enhanced auto-improvement capabilities.

## Overview

The auto-improvement feature uses AI to analyze and improve files in the TARS codebase and documentation. It can run for a specified time limit, during which it will continuously process files and make improvements.

## Enhanced File Prioritization

The auto-improvement feature now uses a sophisticated file prioritization algorithm to determine which files to process first. The algorithm takes into account multiple factors:

### File Type

Different file types are given different base scores:
- F# files (`.fs`): 2.5 (highest priority as they're the core engine)
- C# files (`.cs`): 2.0
- Markdown files (`.md`): 1.0
- Other files: 0.5

### Content Analysis

Files are analyzed for various indicators that suggest they might benefit from improvement:

#### Code Files (C# and F#)
- TODOs, FIXMEs, and other improvement indicators
- Long methods
- Commented-out code
- Magic numbers
- Nested control structures
- Methods with many parameters

#### Documentation Files (Markdown)
- Few headings (might indicate poor structure)
- Code blocks
- TODOs in markdown

### Recency

More recently modified files get higher priority, as they might be actively developed and benefit more from improvements.

### Complexity

Files with higher complexity scores get higher priority, as they might benefit more from improvements.

### Improvement History

Files that have never been improved get a boost in priority. Files that were improved recently get lower priority to avoid repeatedly improving the same files.

## Usage

### Starting Auto-Improvement

```
tarscli auto-improve --time-limit <minutes> --model <model>
```

- `--time-limit`: The time limit in minutes (default: 60)
- `--model`: The model to use for improvements (default: llama3)

### Checking Status

```
tarscli auto-improve --status
```

This command shows the current status of the auto-improvement process, including:
- Whether it's running
- Start time and elapsed time
- Remaining time
- Number of files processed and remaining
- Current file being processed
- Last improved file
- Total number of improvements
- Top priority files
- Recent improvements

### Stopping Auto-Improvement

```
tarscli auto-improve --stop
```

This command stops the auto-improvement process.

## Implementation Details

The auto-improvement feature is implemented in the following classes:

- `AutoImprovementService`: The main service that manages the auto-improvement process
- `FilePriorityScore`: Represents a priority score for a file
- `AutoImprovementState`: Represents the state of the auto-improvement process
- `AutoImprovementStatus`: Represents the status of the auto-improvement process
- `ImprovementRecord`: Represents a record of an improvement made to a file

The file prioritization algorithm is implemented in the `CalculateFilePriorityScore` method of the `AutoImprovementService` class.

## Future Enhancements

Possible future enhancements to the auto-improvement feature include:

- More sophisticated content analysis
- Learning from user feedback
- Multi-file improvements
- Integration with other TARS services
- More detailed reporting and visualization of improvements
