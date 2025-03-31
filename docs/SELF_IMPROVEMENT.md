# TARS Self-Improvement System

This document provides technical details about the TARS self-improvement system, its architecture, and implementation.

## Architecture Overview

The self-improvement system consists of several interconnected components:

1. **Code Analyzer**: Examines code for potential issues and improvement opportunities
2. **Pattern Recognizer**: Identifies specific patterns that could be improved
3. **Improvement Generator**: Creates proposals for code improvements
4. **Code Transformer**: Applies approved improvements to the codebase
5. **Learning Database**: Records improvements and feedback for future learning
6. **Console Capture**: Captures console output and uses it to identify and fix issues

## Implementation Details

### Code Analyzer

The Code Analyzer is implemented in F# and uses a combination of static analysis and LLM-based techniques:

```fsharp
// Simplified example of the analyze function
let analyze (filePath: string) (ollamaEndpoint: string) (model: string) =
    async {
        let! content = File.ReadAllTextAsync(filePath) |> Async.AwaitTask
        let fileType = getFileType filePath

        // Generate prompt for analysis
        let prompt = createAnalysisPrompt filePath content fileType

        // Call Ollama API with proper JSON escaping
        let escapedPrompt = escapeJsonString prompt
        let! response = callOllamaApi ollamaEndpoint model escapedPrompt

        // Parse the response
        return parseAnalysisResponse response
    }
```

### Pattern Recognizer

The Pattern Recognizer identifies common code issues such as:

- Magic numbers (unnamed numeric literals)
- Empty catch blocks
- String concatenation in loops
- Unused variables
- Mutable variables in F# (when immutable alternatives exist)
- Imperative loops in F# (when functional alternatives exist)
- Long methods
- TODO comments

Each pattern has a specific detection algorithm and a corresponding improvement strategy.

### Improvement Generator

The Improvement Generator creates proposals based on the analysis results:

```fsharp
// Simplified example of the improve function
let improve (filePath: string) (ollamaEndpoint: string) (model: string) =
    async {
        let! analysisResult = analyze filePath ollamaEndpoint model

        if analysisResult.Issues.Length > 0 then
            // Generate prompt for improvement
            let prompt = createImprovementPrompt filePath analysisResult

            // Call Ollama API with proper JSON escaping
            let escapedPrompt = escapeJsonString prompt
            let! response = callOllamaApi ollamaEndpoint model escapedPrompt

            // Parse the response and create improvement proposal
            return parseImprovementResponse response filePath
        else
            return None
    }
```

### Code Transformer

The Code Transformer applies the approved improvements to the codebase:

```fsharp
// Simplified example of the apply function
let applyImprovement (proposal: ImprovementProposal) =
    async {
        try
            // Backup the original file
            let backupPath = proposal.FileName + ".bak"
            File.Copy(proposal.FileName, backupPath, true)

            // Write the improved content
            do! File.WriteAllTextAsync(proposal.FileName, proposal.ImprovedContent) |> Async.AwaitTask

            return true
        with ex ->
            // Restore from backup if failed
            // ...
            return false
    }
```

### Learning Database

The Learning Database records improvements and feedback:

```fsharp
// Simplified example of the recordImprovement function
let recordImprovement (filePath: string) (fileType: string) (proposal: ImprovementProposal) =
    async {
        let event = {
            Id = Guid.NewGuid().ToString()
            Timestamp = DateTime.UtcNow
            FilePath = filePath
            FileType = fileType
            ImprovementProposal = Some proposal
            Feedback = None
        }

        do! persistEvent event
        return event.Id
    }
```

### Console Capture

The Console Capture component captures console output and uses it to identify and fix issues:

```csharp
// Simplified example of the ConsoleCaptureService
public class ConsoleCaptureService
{
    private readonly ILogger<ConsoleCaptureService> _logger;
    private readonly OllamaService _ollamaService;
    private readonly McpService _mcpService;
    private TextWriter _originalOut;
    private TextWriter _originalError;
    private MemoryStream _memoryStream;
    private StreamWriter _streamWriter;
    private bool _isCapturing = false;
    private readonly List<string> _capturedOutput = new();

    // Start capturing console output
    public void StartCapture()
    {
        // Save original console output and error writers
        _originalOut = Console.Out;
        _originalError = Console.Error;

        // Create memory stream and writer
        _memoryStream = new MemoryStream();
        _streamWriter = new StreamWriter(_memoryStream) { AutoFlush = true };

        // Redirect console output
        Console.SetOut(_streamWriter);
        Console.SetError(_streamWriter);

        _isCapturing = true;
        _capturedOutput.Clear();
    }

    // Stop capturing and return the captured output
    public string StopCapture()
    {
        // Restore original console output and error writers
        Console.SetOut(_originalOut);
        Console.SetError(_originalError);

        // Get captured output
        _memoryStream.Position = 0;
        using var reader = new StreamReader(_memoryStream);
        var capturedText = reader.ReadToEnd();

        // Clean up
        _streamWriter.Dispose();
        _memoryStream.Dispose();

        _isCapturing = false;
        _capturedOutput.Add(capturedText);

        return capturedText;
    }

    // Analyze captured output and suggest improvements
    public async Task<string> AnalyzeAndSuggestImprovements(string capturedOutput, string filePath)
    {
        // Create prompt for analysis
        var prompt = $"Analyze this code and console output...";

        // Use Ollama to generate suggestions
        return await _ollamaService.GenerateCompletion(prompt, "llama3");
    }
}
```

## CLI Commands

The self-improvement system is accessible through the following CLI commands:

- `self-analyze`: Analyze a file for potential improvements
- `self-propose`: Generate improvement proposals for a file
- `self-rewrite`: Apply improvements to a file
- `self-stats`: Display statistics from the learning database
- `console-capture`: Capture console output and use it to improve code
  - `--start`: Start capturing console output
  - `--stop`: Stop capturing console output
  - `--analyze <file>`: Analyze captured output and suggest improvements
  - `--apply`: Apply suggested improvements
  - `--auto <file>`: Automatically improve code based on captured output

## First Self-Improvement Iteration

On March 29, 2025, we ran the first self-improvement iteration on a test file with intentional code issues:

### Test Code

```csharp
using System;
using System.Collections.Generic;

namespace TestCode
{
    public class Program
    {
        public static void Main(string[] args)
        {
            // This is a test program with some issues to be improved
            Console.WriteLine("Hello, World!");

            // Issue 1: Magic numbers
            int timeout = 300;

            // Issue 2: Inefficient string concatenation in loop
            string result = "";
            for (int i = 0; i < 100; i++)
            {
                result += i.ToString();
            }

            // Issue 3: Empty catch block
            try
            {
                int x = int.Parse("abc");
            }
            catch (Exception)
            {
                // Empty catch block
            }

            // Issue 4: Unused variable
            var unusedList = new List<string>();

            Console.WriteLine(result);
            Console.WriteLine($"Timeout is set to {timeout} seconds");
        }
    }
}
```

### Analysis Results

The analysis successfully identified the following issues:

- Magic numbers on multiple lines (13, 14, 16, 18, 23, 33)
- Inefficient string concatenation in a loop on line 18

And provided these recommendations:

- Replace magic numbers with named constants
- Use StringBuilder instead of string concatenation in loops

### Next Steps

1. Complete the first full self-improvement iteration
2. Enhance the learning database implementation
3. Add more code patterns to the pattern recognition system
4. Improve the quality of explanations in improvement proposals
5. Create visualizations of the self-improvement process
6. Implement metrics to measure improvement quality

## Future Enhancements

- **Multi-file Analysis**: Analyze relationships between files
- **Project-wide Improvements**: Suggest refactorings across multiple files
- **Learning from User Feedback**: Adjust improvement strategies based on user acceptance
- **Language-specific Optimizations**: Tailor improvements to specific programming languages
- **Performance Metrics**: Measure the impact of improvements on code quality and performance
- **Enhanced Console Capture**: Improve pattern recognition for console output
- **CI/CD Integration**: Automatically capture and fix issues during CI/CD pipelines
- **Automated Testing**: Generate and run tests to verify improvements
