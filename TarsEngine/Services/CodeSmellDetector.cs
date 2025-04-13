using System.Text.RegularExpressions;
using Microsoft.Extensions.Logging;
using TarsEngine.Models;

namespace TarsEngine.Services;

/// <summary>
/// Service for detecting code smells in source code
/// </summary>
public class CodeSmellDetector
{
    private readonly ILogger _logger;
    private readonly Dictionary<string, List<Func<string, List<CodeIssue>>>> _languageDetectors = new();

    /// <summary>
    /// Initializes a new instance of the <see cref="CodeSmellDetector"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    public CodeSmellDetector(ILogger logger)
    {
        _logger = logger;
        InitializeDetectors();
    }

    /// <summary>
    /// Detects code smells in the provided content
    /// </summary>
    /// <param name="content">The code content to analyze</param>
    /// <param name="language">The programming language of the code</param>
    /// <returns>The list of detected code smells</returns>
    public List<CodeIssue> DetectCodeSmells(string content, string language)
    {
        try
        {
            _logger.LogInformation("Detecting code smells in {Language} code of length {Length}", language, content?.Length ?? 0);

            var issues = new List<CodeIssue>();

            // Apply common detectors
            issues.AddRange(DetectLongLines(content));
            issues.AddRange(DetectDuplicatedCode(content));
            issues.AddRange(DetectTodoComments(content));
            issues.AddRange(DetectMagicNumbers(content));

            // Apply language-specific detectors
            if (_languageDetectors.TryGetValue(language.ToLowerInvariant(), out var detectors))
            {
                foreach (var detector in detectors)
                {
                    issues.AddRange(detector(content));
                }
            }

            _logger.LogInformation("Detected {IssueCount} code smells in {Language} code", issues.Count, language);
            return issues;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error detecting code smells in {Language} code", language);
            return [];
        }
    }

    private List<CodeIssue> DetectLongLines(string content)
    {
        var issues = new List<CodeIssue>();
        var lines = content.Split('\n');
        const int maxLineLength = 120;

        for (int i = 0; i < lines.Length; i++)
        {
            var line = lines[i];
            if (line.Length > maxLineLength)
            {
                issues.Add(new CodeIssue
                {
                    Type = CodeIssueType.Style,
                    Severity = IssueSeverity.Minor,
                    Title = "Long Line",
                    Description = $"Line exceeds the maximum recommended length of {maxLineLength} characters.",
                    Location = new CodeLocation
                    {
                        StartLine = i,
                        EndLine = i
                    },
                    CodeSnippet = line,
                    SuggestedFix = "Break the line into multiple lines or simplify the expression.",
                    ImpactScore = 0.2,
                    FixDifficultyScore = 0.3,
                    Tags = { "style", "readability" }
                });
            }
        }

        return issues;
    }

    private List<CodeIssue> DetectDuplicatedCode(string content)
    {
        var issues = new List<CodeIssue>();
        var lines = content.Split('\n');
        const int minDuplicateLines = 5;

        // Simple duplication detection for demonstration purposes
        // In a real implementation, this would use a more sophisticated algorithm
        var lineHashes = new Dictionary<string, List<int>>();
        
        for (int i = 0; i < lines.Length; i++)
        {
            var line = lines[i].Trim();
            if (string.IsNullOrWhiteSpace(line) || line.StartsWith("//") || line.StartsWith("/*") || line.StartsWith("*"))
            {
                continue;
            }

            if (!lineHashes.ContainsKey(line))
            {
                lineHashes[line] = [];
            }
            
            lineHashes[line].Add(i);
        }

        // Find sequences of duplicated lines
        var duplicatedSequences = new List<(int Start1, int Start2, int Length)>();
        
        foreach (var lineGroup in lineHashes.Values.Where(v => v.Count > 1))
        {
            for (int i = 0; i < lineGroup.Count - 1; i++)
            {
                for (int j = i + 1; j < lineGroup.Count; j++)
                {
                    int start1 = lineGroup[i];
                    int start2 = lineGroup[j];
                    int length = 1;
                    
                    while (start1 + length < lines.Length && 
                           start2 + length < lines.Length && 
                           lines[start1 + length].Trim() == lines[start2 + length].Trim())
                    {
                        length++;
                    }
                    
                    if (length >= minDuplicateLines)
                    {
                        duplicatedSequences.Add((start1, start2, length));
                    }
                }
            }
        }

        // Create issues for duplicated sequences
        foreach (var (start1, start2, length) in duplicatedSequences)
        {
            var duplicatedCode = string.Join("\n", lines.Skip(start1).Take(length));
            
            issues.Add(new CodeIssue
            {
                Type = CodeIssueType.Duplication,
                Severity = IssueSeverity.Major,
                Title = "Duplicated Code",
                Description = $"Found duplicated code sequence of {length} lines. The same sequence appears at lines {start1 + 1} and {start2 + 1}.",
                Location = new CodeLocation
                {
                    StartLine = start1,
                    EndLine = start1 + length - 1
                },
                CodeSnippet = duplicatedCode.Length > 200 ? duplicatedCode.Substring(0, 200) + "..." : duplicatedCode,
                SuggestedFix = "Extract the duplicated code into a reusable method or function.",
                ImpactScore = 0.7,
                FixDifficultyScore = 0.5,
                Tags = { "duplication", "maintainability" }
            });
        }

        return issues;
    }

    private List<CodeIssue> DetectTodoComments(string content)
    {
        var issues = new List<CodeIssue>();
        var lines = content.Split('\n');
        var todoRegex = new Regex(@"(//|/\*|\*)\s*(TODO|FIXME|HACK|XXX):", RegexOptions.IgnoreCase | RegexOptions.Compiled);

        for (int i = 0; i < lines.Length; i++)
        {
            var match = todoRegex.Match(lines[i]);
            if (match.Success)
            {
                var todoType = match.Groups[2].Value.ToUpper();
                var severity = todoType switch
                {
                    "FIXME" => IssueSeverity.Major,
                    "HACK" => IssueSeverity.Major,
                    "XXX" => IssueSeverity.Major,
                    _ => IssueSeverity.Minor
                };

                issues.Add(new CodeIssue
                {
                    Type = CodeIssueType.Documentation,
                    Severity = severity,
                    Title = $"{todoType} Comment",
                    Description = $"Found {todoType} comment that needs to be addressed.",
                    Location = new CodeLocation
                    {
                        StartLine = i,
                        EndLine = i
                    },
                    CodeSnippet = lines[i],
                    SuggestedFix = $"Address the {todoType} comment or convert it to a tracked issue.",
                    ImpactScore = todoType == "TODO" ? 0.3 : 0.6,
                    FixDifficultyScore = 0.5,
                    Tags = { "documentation", "technical-debt" }
                });
            }
        }

        return issues;
    }

    private List<CodeIssue> DetectMagicNumbers(string content)
    {
        var issues = new List<CodeIssue>();
        var lines = content.Split('\n');
        
        // Regex to find magic numbers, excluding common cases like 0, 1, -1
        var magicNumberRegex = new Regex(@"[^.\w](-?\d+\.?\d*)[^.\w]", RegexOptions.Compiled);
        var allowedNumbers = new HashSet<string> { "0", "1", "-1", "2", "100", "1000" };

        for (int i = 0; i < lines.Length; i++)
        {
            var line = lines[i];
            
            // Skip comments and string literals
            if (line.Trim().StartsWith("//") || line.Contains("\""))
            {
                continue;
            }
            
            var matches = magicNumberRegex.Matches(line);
            foreach (Match match in matches)
            {
                var number = match.Groups[1].Value;
                if (!allowedNumbers.Contains(number))
                {
                    issues.Add(new CodeIssue
                    {
                        Type = CodeIssueType.Maintainability,
                        Severity = IssueSeverity.Minor,
                        Title = "Magic Number",
                        Description = $"Found magic number '{number}' that should be replaced with a named constant.",
                        Location = new CodeLocation
                        {
                            StartLine = i,
                            EndLine = i
                        },
                        CodeSnippet = line,
                        SuggestedFix = $"Replace '{number}' with a named constant to improve code readability and maintainability.",
                        ImpactScore = 0.4,
                        FixDifficultyScore = 0.2,
                        Tags = { "maintainability", "readability" }
                    });
                }
            }
        }

        return issues;
    }

    private List<CodeIssue> DetectCSharpCodeSmells(string content)
    {
        var issues = new List<CodeIssue>();
        var lines = content.Split('\n');

        // Detect large classes
        var classRegex = new Regex(@"(public|private|protected|internal)?\s*(static|abstract|sealed)?\s*class\s+([a-zA-Z0-9_]+)", RegexOptions.Compiled);
        var classMatches = classRegex.Matches(content);
        
        foreach (Match match in classMatches)
        {
            var className = match.Groups[3].Value;
            var startLine = content.Substring(0, match.Index).Count(c => c == '\n');
            
            // Find the end of the class (simplified approach)
            var classContent = content.Substring(match.Index);
            var braceCount = 0;
            var endIndex = 0;
            
            for (int i = 0; i < classContent.Length; i++)
            {
                if (classContent[i] == '{')
                {
                    braceCount++;
                }
                else if (classContent[i] == '}')
                {
                    braceCount--;
                    if (braceCount == 0)
                    {
                        endIndex = i;
                        break;
                    }
                }
            }
            
            if (endIndex > 0)
            {
                var classLines = classContent.Substring(0, endIndex).Count(c => c == '\n') + 1;
                
                if (classLines > 300)
                {
                    issues.Add(new CodeIssue
                    {
                        Type = CodeIssueType.Maintainability,
                        Severity = IssueSeverity.Major,
                        Title = "Large Class",
                        Description = $"Class '{className}' has {classLines} lines, which exceeds the recommended maximum of 300 lines.",
                        Location = new CodeLocation
                        {
                            StartLine = startLine,
                            EndLine = startLine + classLines
                        },
                        SuggestedFix = "Consider breaking the class into smaller, more focused classes.",
                        ImpactScore = 0.7,
                        FixDifficultyScore = 0.7,
                        Tags = { "maintainability", "large-class" }
                    });
                }
            }
        }

        // Detect long methods
        var methodRegex = new Regex(@"(public|private|protected|internal)?\s*(static|virtual|abstract|override|async)?\s*([a-zA-Z0-9_<>]+)\s+([a-zA-Z0-9_]+)\s*\(", RegexOptions.Compiled);
        var methodMatches = methodRegex.Matches(content);
        
        foreach (Match match in methodMatches)
        {
            var methodName = match.Groups[4].Value;
            var startLine = content.Substring(0, match.Index).Count(c => c == '\n');
            
            // Find the end of the method (simplified approach)
            var methodContent = content.Substring(match.Index);
            var braceCount = 0;
            var inMethod = false;
            var endIndex = 0;
            
            for (int i = 0; i < methodContent.Length; i++)
            {
                if (methodContent[i] == '{')
                {
                    if (!inMethod)
                    {
                        inMethod = true;
                    }
                    braceCount++;
                }
                else if (methodContent[i] == '}')
                {
                    braceCount--;
                    if (inMethod && braceCount == 0)
                    {
                        endIndex = i;
                        break;
                    }
                }
            }
            
            if (endIndex > 0)
            {
                var methodLines = methodContent.Substring(0, endIndex).Count(c => c == '\n') + 1;
                
                if (methodLines > 30)
                {
                    issues.Add(new CodeIssue
                    {
                        Type = CodeIssueType.Maintainability,
                        Severity = IssueSeverity.Major,
                        Title = "Long Method",
                        Description = $"Method '{methodName}' has {methodLines} lines, which exceeds the recommended maximum of 30 lines.",
                        Location = new CodeLocation
                        {
                            StartLine = startLine,
                            EndLine = startLine + methodLines
                        },
                        SuggestedFix = "Consider breaking the method into smaller, more focused methods.",
                        ImpactScore = 0.6,
                        FixDifficultyScore = 0.5,
                        Tags = { "maintainability", "long-method" }
                    });
                }
            }
        }

        // Detect unused using directives
        var usingRegex = new Regex(@"using\s+([a-zA-Z0-9_.]+);", RegexOptions.Compiled);
        var usingMatches = usingRegex.Matches(content);
        var usings = new List<(string Namespace, int Line)>();
        
        foreach (Match match in usingMatches)
        {
            var ns = match.Groups[1].Value;
            var line = content.Substring(0, match.Index).Count(c => c == '\n');
            usings.Add((ns, line));
        }
        
        // Simple check for unused usings (would be more sophisticated in a real implementation)
        foreach (var (ns, line) in usings)
        {
            var nsWithoutSystem = ns.Replace("System.", "");
            var contentAfterUsings = content.Substring(content.LastIndexOf("using", StringComparison.Ordinal) + 5);
            
            if (!contentAfterUsings.Contains(nsWithoutSystem) && !ns.StartsWith("System"))
            {
                issues.Add(new CodeIssue
                {
                    Type = CodeIssueType.CodeSmell,
                    Severity = IssueSeverity.Minor,
                    Title = "Unused Using Directive",
                    Description = $"The using directive for namespace '{ns}' appears to be unused.",
                    Location = new CodeLocation
                    {
                        StartLine = line,
                        EndLine = line
                    },
                    SuggestedFix = $"Remove the unused using directive for '{ns}'.",
                    ImpactScore = 0.2,
                    FixDifficultyScore = 0.1,
                    Tags = { "maintainability", "unused-code" }
                });
            }
        }

        return issues;
    }

    private List<CodeIssue> DetectFSharpCodeSmells(string content)
    {
        var issues = new List<CodeIssue>();
        var lines = content.Split('\n');

        // Detect mutable variables
        var mutableRegex = new Regex(@"let\s+mutable\s+([a-zA-Z0-9_]+)", RegexOptions.Compiled);
        var mutableMatches = mutableRegex.Matches(content);
        
        foreach (Match match in mutableMatches)
        {
            var variableName = match.Groups[1].Value;
            var line = content.Substring(0, match.Index).Count(c => c == '\n');
            
            issues.Add(new CodeIssue
            {
                Type = CodeIssueType.CodeSmell,
                Severity = IssueSeverity.Minor,
                Title = "Mutable Variable",
                Description = $"The variable '{variableName}' is declared as mutable. Consider using immutable values instead.",
                Location = new CodeLocation
                {
                    StartLine = line,
                    EndLine = line
                },
                SuggestedFix = "Refactor the code to use immutable values and functional transformations.",
                ImpactScore = 0.4,
                FixDifficultyScore = 0.5,
                Tags = { "functional-style", "immutability" }
            });
        }

        // Detect long functions
        var functionRegex = new Regex(@"let\s+([a-zA-Z0-9_]+)(?:\s+[a-zA-Z0-9_]+)*\s*=", RegexOptions.Compiled);
        var functionMatches = functionRegex.Matches(content);
        
        foreach (Match match in functionMatches)
        {
            var functionName = match.Groups[1].Value;
            var startLine = content.Substring(0, match.Index).Count(c => c == '\n');
            
            // Find the end of the function (simplified approach)
            var endLine = startLine;
            for (int i = startLine + 1; i < lines.Length; i++)
            {
                if (lines[i].Trim().StartsWith("let ") || lines[i].Trim().StartsWith("type ") || lines[i].Trim().StartsWith("module "))
                {
                    endLine = i - 1;
                    break;
                }
                
                if (i == lines.Length - 1)
                {
                    endLine = i;
                }
            }
            
            var functionLines = endLine - startLine + 1;
            
            if (functionLines > 30)
            {
                issues.Add(new CodeIssue
                {
                    Type = CodeIssueType.Maintainability,
                    Severity = IssueSeverity.Major,
                    Title = "Long Function",
                    Description = $"Function '{functionName}' has {functionLines} lines, which exceeds the recommended maximum of 30 lines.",
                    Location = new CodeLocation
                    {
                        StartLine = startLine,
                        EndLine = endLine
                    },
                    SuggestedFix = "Consider breaking the function into smaller, more focused functions.",
                    ImpactScore = 0.6,
                    FixDifficultyScore = 0.5,
                    Tags = { "maintainability", "long-function" }
                });
            }
        }

        // Detect non-idiomatic F# code (using imperative loops instead of higher-order functions)
        var forLoopRegex = new Regex(@"for\s+[a-zA-Z0-9_]+\s+in", RegexOptions.Compiled);
        var forLoopMatches = forLoopRegex.Matches(content);
        
        foreach (Match match in forLoopMatches)
        {
            var line = content.Substring(0, match.Index).Count(c => c == '\n');
            
            issues.Add(new CodeIssue
            {
                Type = CodeIssueType.Style,
                Severity = IssueSeverity.Minor,
                Title = "Imperative Loop",
                Description = "Using imperative for loop instead of higher-order functions like map, filter, or fold.",
                Location = new CodeLocation
                {
                    StartLine = line,
                    EndLine = line
                },
                SuggestedFix = "Consider using higher-order functions like List.map, List.filter, or List.fold instead of imperative loops.",
                ImpactScore = 0.3,
                FixDifficultyScore = 0.4,
                Tags = { "functional-style", "idiomatic-fsharp" }
            });
        }

        return issues;
    }

    private void InitializeDetectors()
    {
        // Register C# detectors
        _languageDetectors["csharp"] = [DetectCSharpCodeSmells];

        // Register F# detectors
        _languageDetectors["fsharp"] = [DetectFSharpCodeSmells];

        // Add more language-specific detectors as needed
    }
}
