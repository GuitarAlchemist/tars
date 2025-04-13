using System.Text.RegularExpressions;
using Microsoft.Extensions.Logging;
using TarsEngine.Models;

namespace TarsEngine.Services;

/// <summary>
/// Service for analyzing code for performance issues
/// </summary>
public class PerformanceAnalyzer
{
    private readonly ILogger _logger;
    private readonly Dictionary<string, List<Func<string, List<CodeIssue>>>> _languageDetectors = new();

    /// <summary>
    /// Initializes a new instance of the <see cref="PerformanceAnalyzer"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    public PerformanceAnalyzer(ILogger logger)
    {
        _logger = logger;
        InitializeDetectors();
    }

    /// <summary>
    /// Detects performance issues in the provided content
    /// </summary>
    /// <param name="content">The code content to analyze</param>
    /// <param name="language">The programming language of the code</param>
    /// <returns>The list of detected performance issues</returns>
    public List<CodeIssue> DetectPerformanceIssues(string content, string language)
    {
        try
        {
            _logger.LogInformation("Detecting performance issues in {Language} code of length {Length}", language, content?.Length ?? 0);

            var issues = new List<CodeIssue>();

            // Apply common detectors
            issues.AddRange(DetectLargeObjectCreation(content));
            issues.AddRange(DetectStringConcatenationInLoops(content));
            issues.AddRange(DetectNestedLoops(content));

            // Apply language-specific detectors
            if (_languageDetectors.TryGetValue(language.ToLowerInvariant(), out var detectors))
            {
                foreach (var detector in detectors)
                {
                    issues.AddRange(detector(content));
                }
            }

            _logger.LogInformation("Detected {IssueCount} performance issues in {Language} code", issues.Count, language);
            return issues;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error detecting performance issues in {Language} code", language);
            return [];
        }
    }

    private List<CodeIssue> DetectLargeObjectCreation(string content)
    {
        var issues = new List<CodeIssue>();
        var lines = content.Split('\n');

        // Look for large array or collection initializations
        var arrayInitRegex = new Regex(@"new\s+[a-zA-Z0-9_<>]+\s*\[\s*(\d+)\s*\]", RegexOptions.Compiled);
        
        for (int i = 0; i < lines.Length; i++)
        {
            var line = lines[i];
            var matches = arrayInitRegex.Matches(line);
            
            foreach (Match match in matches)
            {
                if (match.Groups.Count > 1 && int.TryParse(match.Groups[1].Value, out var size))
                {
                    if (size > 10000)
                    {
                        issues.Add(new CodeIssue
                        {
                            Type = CodeIssueType.Performance,
                            Severity = size > 1000000 ? IssueSeverity.Critical : (size > 100000 ? IssueSeverity.Major : IssueSeverity.Minor),
                            Title = "Large Array Allocation",
                            Description = $"Creating a large array with {size} elements may cause performance issues or memory pressure.",
                            Location = new CodeLocation
                            {
                                StartLine = i,
                                EndLine = i
                            },
                            CodeSnippet = line,
                            SuggestedFix = "Consider using a more memory-efficient data structure or lazy initialization.",
                            ImpactScore = Math.Min(1.0, Math.Log10(size) / 7),
                            FixDifficultyScore = 0.4,
                            Tags = { "performance", "memory" }
                        });
                    }
                }
            }
        }

        return issues;
    }

    private List<CodeIssue> DetectStringConcatenationInLoops(string content)
    {
        var issues = new List<CodeIssue>();
        var lines = content.Split('\n');
        
        // Find loops
        var loopStartRegex = new Regex(@"\b(for|foreach|while)\b", RegexOptions.Compiled);
        var stringConcatRegex = new Regex(@"[a-zA-Z0-9_]+\s*\+=\s*[""']|[a-zA-Z0-9_]+\s*=\s*[a-zA-Z0-9_]+\s*\+\s*[""']", RegexOptions.Compiled);
        
        int loopStartLine = -1;
        int braceCount = 0;
        bool inLoop = false;
        
        for (int i = 0; i < lines.Length; i++)
        {
            var line = lines[i];
            
            // Check if this line starts a loop
            if (!inLoop && loopStartRegex.IsMatch(line))
            {
                loopStartLine = i;
                inLoop = true;
                if (line.Contains("{"))
                {
                    braceCount++;
                }
                continue;
            }
            
            // Update brace count
            if (inLoop)
            {
                braceCount += line.Count(c => c == '{');
                braceCount -= line.Count(c => c == '}');
                
                // Check for string concatenation in the loop
                if (stringConcatRegex.IsMatch(line))
                {
                    issues.Add(new CodeIssue
                    {
                        Type = CodeIssueType.Performance,
                        Severity = IssueSeverity.Major,
                        Title = "String Concatenation in Loop",
                        Description = "String concatenation inside a loop can lead to performance issues due to the immutable nature of strings.",
                        Location = new CodeLocation
                        {
                            StartLine = i,
                            EndLine = i
                        },
                        CodeSnippet = line,
                        SuggestedFix = "Use StringBuilder instead of string concatenation in loops.",
                        ImpactScore = 0.7,
                        FixDifficultyScore = 0.3,
                        Tags = { "performance", "string-concatenation" }
                    });
                }
                
                // Check if we've exited the loop
                if (braceCount <= 0)
                {
                    inLoop = false;
                    loopStartLine = -1;
                }
            }
        }
        
        return issues;
    }

    private List<CodeIssue> DetectNestedLoops(string content)
    {
        var issues = new List<CodeIssue>();
        var lines = content.Split('\n');
        
        // Find nested loops
        var loopRegex = new Regex(@"\b(for|foreach|while)\b", RegexOptions.Compiled);
        var loopStack = new Stack<int>();
        var braceStack = new Stack<int>();
        
        for (int i = 0; i < lines.Length; i++)
        {
            var line = lines[i];
            
            // Count opening and closing braces
            var openBraces = line.Count(c => c == '{');
            var closeBraces = line.Count(c => c == '}');
            
            // Update brace stack
            for (int j = 0; j < openBraces; j++)
            {
                braceStack.Push(i);
            }
            
            for (int j = 0; j < closeBraces; j++)
            {
                if (braceStack.Count > 0)
                {
                    var openLine = braceStack.Pop();
                    
                    // If this brace closes a loop, pop from loop stack
                    if (loopStack.Count > 0 && loopStack.Peek() <= openLine)
                    {
                        loopStack.Pop();
                    }
                }
            }
            
            // Check if this line starts a loop
            if (loopRegex.IsMatch(line))
            {
                // If we already have loops in the stack, this is a nested loop
                if (loopStack.Count >= 2)
                {
                    issues.Add(new CodeIssue
                    {
                        Type = CodeIssueType.Performance,
                        Severity = loopStack.Count >= 3 ? IssueSeverity.Critical : IssueSeverity.Major,
                        Title = $"Nested Loop (Depth {loopStack.Count + 1})",
                        Description = $"Nested loops with depth {loopStack.Count + 1} can lead to performance issues with O(n^{loopStack.Count + 1}) complexity.",
                        Location = new CodeLocation
                        {
                            StartLine = i,
                            EndLine = i
                        },
                        CodeSnippet = line,
                        SuggestedFix = "Consider refactoring to reduce loop nesting or use more efficient algorithms.",
                        ImpactScore = Math.Min(1.0, (loopStack.Count + 1) / 4.0),
                        FixDifficultyScore = Math.Min(1.0, (loopStack.Count + 1) / 3.0),
                        Tags = { "performance", "algorithm-complexity" }
                    });
                }
                
                // Add this loop to the stack
                loopStack.Push(i);
            }
        }
        
        return issues;
    }

    private List<CodeIssue> DetectCSharpPerformanceIssues(string content)
    {
        var issues = new List<CodeIssue>();
        var lines = content.Split('\n');

        // Detect inefficient LINQ queries
        issues.AddRange(DetectInefficientLinqQueries(content, lines));
        
        // Detect boxing/unboxing
        issues.AddRange(DetectBoxingUnboxing(content, lines));
        
        // Detect inefficient exception handling
        issues.AddRange(DetectInefficientExceptionHandling(content, lines));
        
        // Detect unnecessary object creation
        issues.AddRange(DetectUnnecessaryObjectCreation(content, lines));

        return issues;
    }

    private List<CodeIssue> DetectInefficientLinqQueries(string content, string[] lines)
    {
        var issues = new List<CodeIssue>();
        
        // Look for multiple LINQ operations that could be combined
        var linqOperationRegex = new Regex(@"\.(Where|Select|OrderBy|OrderByDescending|GroupBy|Join|Skip|Take)\(", RegexOptions.Compiled);
        
        for (int i = 0; i < lines.Length; i++)
        {
            var line = lines[i];
            var matches = linqOperationRegex.Matches(line);
            
            if (matches.Count >= 3)
            {
                issues.Add(new CodeIssue
                {
                    Type = CodeIssueType.Performance,
                    Severity = IssueSeverity.Minor,
                    Title = "Multiple LINQ Operations",
                    Description = $"Multiple chained LINQ operations may cause multiple iterations over the collection.",
                    Location = new CodeLocation
                    {
                        StartLine = i,
                        EndLine = i
                    },
                    CodeSnippet = line,
                    SuggestedFix = "Consider combining operations or using more efficient approaches like query comprehension syntax.",
                    ImpactScore = 0.5,
                    FixDifficultyScore = 0.4,
                    Tags = { "performance", "linq" }
                });
            }
        }
        
        // Look for ToList/ToArray followed by LINQ operations
        var materializeRegex = new Regex(@"\.(ToList|ToArray)\(\)\s*\.\s*(Where|Select|OrderBy|OrderByDescending|GroupBy|Join|Skip|Take)\(", RegexOptions.Compiled);
        
        for (int i = 0; i < lines.Length; i++)
        {
            var line = lines[i];
            var matches = materializeRegex.Matches(line);
            
            if (matches.Count > 0)
            {
                issues.Add(new CodeIssue
                {
                    Type = CodeIssueType.Performance,
                    Severity = IssueSeverity.Minor,
                    Title = "Inefficient LINQ Operation Order",
                    Description = "Converting to List/Array before performing LINQ operations causes unnecessary materialization.",
                    Location = new CodeLocation
                    {
                        StartLine = i,
                        EndLine = i
                    },
                    CodeSnippet = line,
                    SuggestedFix = "Perform LINQ operations before converting to List/Array.",
                    ImpactScore = 0.6,
                    FixDifficultyScore = 0.2,
                    Tags = { "performance", "linq" }
                });
            }
        }
        
        return issues;
    }

    private List<CodeIssue> DetectBoxingUnboxing(string content, string[] lines)
    {
        var issues = new List<CodeIssue>();
        
        // Look for boxing operations (simplified detection)
        var boxingRegex = new Regex(@"(int|double|float|bool|char|byte|short|long|decimal)\s+[a-zA-Z0-9_]+\s*=\s*[^;]+;\s*object\s+[a-zA-Z0-9_]+\s*=\s*[a-zA-Z0-9_]+;", RegexOptions.Compiled);
        var matches = boxingRegex.Matches(content);
        
        foreach (Match match in matches)
        {
            var lineNumber = content.Substring(0, match.Index).Count(c => c == '\n');
            
            issues.Add(new CodeIssue
            {
                Type = CodeIssueType.Performance,
                Severity = IssueSeverity.Minor,
                Title = "Value Type Boxing",
                Description = "Boxing of value types can lead to performance issues and increased memory usage.",
                Location = new CodeLocation
                {
                    StartLine = lineNumber,
                    EndLine = lineNumber
                },
                CodeSnippet = match.Value,
                SuggestedFix = "Consider using generics or avoiding object type when working with value types.",
                ImpactScore = 0.4,
                FixDifficultyScore = 0.3,
                Tags = { "performance", "boxing" }
            });
        }
        
        // Look for unboxing operations (simplified detection)
        var unboxingRegex = new Regex(@"\(\s*(int|double|float|bool|char|byte|short|long|decimal)\s*\)\s*[a-zA-Z0-9_]+", RegexOptions.Compiled);
        
        for (int i = 0; i < lines.Length; i++)
        {
            var line = lines[i];
            var unboxingMatches = unboxingRegex.Matches(line);
            
            if (unboxingMatches.Count > 0)
            {
                issues.Add(new CodeIssue
                {
                    Type = CodeIssueType.Performance,
                    Severity = IssueSeverity.Minor,
                    Title = "Value Type Unboxing",
                    Description = "Unboxing of value types can lead to performance issues and potential runtime exceptions.",
                    Location = new CodeLocation
                    {
                        StartLine = i,
                        EndLine = i
                    },
                    CodeSnippet = line,
                    SuggestedFix = "Consider using generics or avoiding object type when working with value types.",
                    ImpactScore = 0.4,
                    FixDifficultyScore = 0.3,
                    Tags = { "performance", "unboxing" }
                });
            }
        }
        
        return issues;
    }

    private List<CodeIssue> DetectInefficientExceptionHandling(string content, string[] lines)
    {
        var issues = new List<CodeIssue>();
        
        // Look for empty catch blocks
        var emptyCatchRegex = new Regex(@"catch\s*\([^)]*\)\s*{\s*}", RegexOptions.Compiled);
        var matches = emptyCatchRegex.Matches(content);
        
        foreach (Match match in matches)
        {
            var lineNumber = content.Substring(0, match.Index).Count(c => c == '\n');
            
            issues.Add(new CodeIssue
            {
                Type = CodeIssueType.Performance,
                Severity = IssueSeverity.Major,
                Title = "Empty Catch Block",
                Description = "Empty catch blocks swallow exceptions and can hide serious issues.",
                Location = new CodeLocation
                {
                    StartLine = lineNumber,
                    EndLine = lineNumber
                },
                CodeSnippet = match.Value,
                SuggestedFix = "Either handle the exception properly or log it at minimum.",
                ImpactScore = 0.7,
                FixDifficultyScore = 0.3,
                Tags = { "performance", "exception-handling" }
            });
        }
        
        // Look for catching generic Exception
        var genericExceptionRegex = new Regex(@"catch\s*\(\s*Exception\s+[a-zA-Z0-9_]+\s*\)", RegexOptions.Compiled);
        
        for (int i = 0; i < lines.Length; i++)
        {
            var line = lines[i];
            var exceptionMatches = genericExceptionRegex.Matches(line);
            
            if (exceptionMatches.Count > 0)
            {
                issues.Add(new CodeIssue
                {
                    Type = CodeIssueType.Performance,
                    Severity = IssueSeverity.Minor,
                    Title = "Catching Generic Exception",
                    Description = "Catching generic Exception can mask unexpected errors and lead to performance issues.",
                    Location = new CodeLocation
                    {
                        StartLine = i,
                        EndLine = i
                    },
                    CodeSnippet = line,
                    SuggestedFix = "Catch specific exception types that you expect and can handle properly.",
                    ImpactScore = 0.5,
                    FixDifficultyScore = 0.3,
                    Tags = { "performance", "exception-handling" }
                });
            }
        }
        
        return issues;
    }

    private List<CodeIssue> DetectUnnecessaryObjectCreation(string content, string[] lines)
    {
        var issues = new List<CodeIssue>();
        
        // Look for new object creation in loops
        var loopStartRegex = new Regex(@"\b(for|foreach|while)\b", RegexOptions.Compiled);
        var objectCreationRegex = new Regex(@"new\s+[a-zA-Z0-9_<>]+", RegexOptions.Compiled);
        
        int loopStartLine = -1;
        int braceCount = 0;
        bool inLoop = false;
        var objectsCreatedInLoop = new HashSet<string>();
        
        for (int i = 0; i < lines.Length; i++)
        {
            var line = lines[i];
            
            // Check if this line starts a loop
            if (!inLoop && loopStartRegex.IsMatch(line))
            {
                loopStartLine = i;
                inLoop = true;
                objectsCreatedInLoop.Clear();
                if (line.Contains("{"))
                {
                    braceCount++;
                }
                continue;
            }
            
            // Update brace count
            if (inLoop)
            {
                braceCount += line.Count(c => c == '{');
                braceCount -= line.Count(c => c == '}');
                
                // Check for object creation in the loop
                var matches = objectCreationRegex.Matches(line);
                foreach (Match match in matches)
                {
                    var objectType = match.Value.Substring(4).Trim();
                    
                    // Skip common types that are typically created in loops
                    if (objectType.StartsWith("KeyValuePair<") || 
                        objectType.StartsWith("Tuple<") || 
                        objectType.StartsWith("ValueTuple<") ||
                        objectType == "StringBuilder" ||
                        objectType == "List<" ||
                        objectType == "Dictionary<")
                    {
                        continue;
                    }
                    
                    if (!objectsCreatedInLoop.Contains(objectType))
                    {
                        objectsCreatedInLoop.Add(objectType);
                    }
                    else
                    {
                        issues.Add(new CodeIssue
                        {
                            Type = CodeIssueType.Performance,
                            Severity = IssueSeverity.Minor,
                            Title = "Repeated Object Creation in Loop",
                            Description = $"Creating new instances of {objectType} repeatedly in a loop can lead to performance issues.",
                            Location = new CodeLocation
                            {
                                StartLine = i,
                                EndLine = i
                            },
                            CodeSnippet = line,
                            SuggestedFix = "Consider creating the object outside the loop and reusing it, or using object pooling.",
                            ImpactScore = 0.5,
                            FixDifficultyScore = 0.4,
                            Tags = { "performance", "object-creation" }
                        });
                    }
                }
                
                // Check if we've exited the loop
                if (braceCount <= 0)
                {
                    inLoop = false;
                    loopStartLine = -1;
                }
            }
        }
        
        return issues;
    }

    private List<CodeIssue> DetectFSharpPerformanceIssues(string content)
    {
        var issues = new List<CodeIssue>();
        var lines = content.Split('\n');

        // Detect inefficient list operations
        issues.AddRange(DetectInefficientListOperations(content, lines));
        
        // Detect excessive recursion
        issues.AddRange(DetectExcessiveRecursion(content, lines));
        
        // Detect inefficient pattern matching
        issues.AddRange(DetectInefficientPatternMatching(content, lines));

        return issues;
    }

    private List<CodeIssue> DetectInefficientListOperations(string content, string[] lines)
    {
        var issues = new List<CodeIssue>();
        
        // Look for list concatenation in loops
        var loopStartRegex = new Regex(@"\bfor\b|\bwhile\b", RegexOptions.Compiled);
        var listConcatRegex = new Regex(@"@|\+\+|List\.append", RegexOptions.Compiled);
        
        int loopStartLine = -1;
        int braceCount = 0;
        bool inLoop = false;
        
        for (int i = 0; i < lines.Length; i++)
        {
            var line = lines[i];
            
            // Check if this line starts a loop
            if (!inLoop && loopStartRegex.IsMatch(line))
            {
                loopStartLine = i;
                inLoop = true;
                if (line.Contains("do"))
                {
                    braceCount++;
                }
                continue;
            }
            
            // Update brace count
            if (inLoop)
            {
                braceCount += line.Count(c => c == 'd' && line.Contains("do"));
                braceCount -= line.Count(c => c == 'd' && line.Contains("done"));
                
                // Check for list concatenation in the loop
                if (listConcatRegex.IsMatch(line))
                {
                    issues.Add(new CodeIssue
                    {
                        Type = CodeIssueType.Performance,
                        Severity = IssueSeverity.Major,
                        Title = "List Concatenation in Loop",
                        Description = "List concatenation inside a loop can lead to performance issues due to the immutable nature of F# lists.",
                        Location = new CodeLocation
                        {
                            StartLine = i,
                            EndLine = i
                        },
                        CodeSnippet = line,
                        SuggestedFix = "Use a mutable collection like ResizeArray (List<T>) during the loop and convert to an immutable list afterward.",
                        ImpactScore = 0.7,
                        FixDifficultyScore = 0.4,
                        Tags = { "performance", "list-concatenation" }
                    });
                }
                
                // Check if we've exited the loop
                if (braceCount <= 0)
                {
                    inLoop = false;
                    loopStartLine = -1;
                }
            }
        }
        
        return issues;
    }

    private List<CodeIssue> DetectExcessiveRecursion(string content, string[] lines)
    {
        var issues = new List<CodeIssue>();
        
        // Look for recursive functions without tail recursion
        var recursiveFunctionRegex = new Regex(@"let\s+rec\s+([a-zA-Z0-9_]+)", RegexOptions.Compiled);
        var matches = recursiveFunctionRegex.Matches(content);
        
        foreach (Match match in matches)
        {
            if (match.Groups.Count > 1)
            {
                var functionName = match.Groups[1].Value;
                var lineNumber = content.Substring(0, match.Index).Count(c => c == '\n');
                
                // Check if the function calls itself at the end (tail recursion)
                var functionContent = ExtractFunctionContent(content, lineNumber);
                var isTailRecursive = IsTailRecursive(functionContent, functionName);
                
                if (!isTailRecursive && functionContent.Contains(functionName))
                {
                    issues.Add(new CodeIssue
                    {
                        Type = CodeIssueType.Performance,
                        Severity = IssueSeverity.Major,
                        Title = "Non-Tail Recursive Function",
                        Description = $"The recursive function '{functionName}' is not tail recursive, which can lead to stack overflow for large inputs.",
                        Location = new CodeLocation
                        {
                            StartLine = lineNumber,
                            EndLine = lineNumber
                        },
                        CodeSnippet = match.Value,
                        SuggestedFix = "Refactor the function to use tail recursion with an accumulator parameter.",
                        ImpactScore = 0.7,
                        FixDifficultyScore = 0.6,
                        Tags = { "performance", "recursion" }
                    });
                }
            }
        }
        
        return issues;
    }

    private List<CodeIssue> DetectInefficientPatternMatching(string content, string[] lines)
    {
        var issues = new List<CodeIssue>();
        
        // Look for nested pattern matching
        var matchRegex = new Regex(@"\bmatch\b", RegexOptions.Compiled);
        var withRegex = new Regex(@"\bwith\b", RegexOptions.Compiled);
        
        int matchStartLine = -1;
        int matchCount = 0;
        
        for (int i = 0; i < lines.Length; i++)
        {
            var line = lines[i];
            
            if (matchRegex.IsMatch(line))
            {
                if (matchStartLine == -1)
                {
                    matchStartLine = i;
                }
                matchCount++;
            }
            
            if (withRegex.IsMatch(line) && matchCount > 0)
            {
                matchCount--;
                
                if (matchCount == 0)
                {
                    matchStartLine = -1;
                }
            }
            
            // If we have nested matches, report an issue
            if (matchCount > 1)
            {
                issues.Add(new CodeIssue
                {
                    Type = CodeIssueType.Performance,
                    Severity = IssueSeverity.Minor,
                    Title = "Nested Pattern Matching",
                    Description = "Nested pattern matching can lead to complex and inefficient code.",
                    Location = new CodeLocation
                    {
                        StartLine = i,
                        EndLine = i
                    },
                    CodeSnippet = line,
                    SuggestedFix = "Consider refactoring to use active patterns or separate functions for each level of matching.",
                    ImpactScore = 0.5,
                    FixDifficultyScore = 0.5,
                    Tags = { "performance", "pattern-matching" }
                });
                
                // Reset to avoid reporting the same nested match multiple times
                matchCount = 1;
            }
        }
        
        return issues;
    }

    private string ExtractFunctionContent(string content, int startLine)
    {
        var lines = content.Split('\n');
        var functionContent = new List<string>();
        
        // Start from the function definition
        for (int i = startLine; i < lines.Length; i++)
        {
            functionContent.Add(lines[i]);
            
            // Stop when we reach the next function or module definition
            if (i > startLine && (lines[i].Trim().StartsWith("let ") || lines[i].Trim().StartsWith("module ")))
            {
                break;
            }
        }
        
        return string.Join("\n", functionContent);
    }

    private bool IsTailRecursive(string functionContent, string functionName)
    {
        // This is a simplified check for tail recursion
        // A more accurate check would parse the AST
        
        var lines = functionContent.Split('\n');
        
        for (int i = lines.Length - 1; i >= 0; i--)
        {
            var line = lines[i].Trim();
            
            // Skip empty lines and comments
            if (string.IsNullOrWhiteSpace(line) || line.StartsWith("//") || line.StartsWith("(*"))
            {
                continue;
            }
            
            // If the last non-empty line contains the function name followed by arguments, it might be tail recursive
            if (line.Contains(functionName) && !line.Contains("let " + functionName))
            {
                return true;
            }
            
            // If we find any other code, it's not tail recursive
            return false;
        }
        
        return false;
    }

    private void InitializeDetectors()
    {
        // Register C# detectors
        _languageDetectors["csharp"] = [DetectCSharpPerformanceIssues];

        // Register F# detectors
        _languageDetectors["fsharp"] = [DetectFSharpPerformanceIssues];

        // Add more language-specific detectors as needed
    }
}
