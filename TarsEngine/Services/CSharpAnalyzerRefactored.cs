using System.Text.RegularExpressions;
using Microsoft.Extensions.Logging;
using TarsEngine.Models;
using TarsEngine.Monads;
using TarsEngine.Services.Interfaces;

namespace TarsEngine.Services;

/// <summary>
/// Analyzer for C# code
/// </summary>
public class CSharpAnalyzerRefactored(ILogger<CSharpAnalyzerRefactored> logger)
    : ILanguageAnalyzer
{
    private readonly ILogger<CSharpAnalyzerRefactored> _logger = logger;
    private readonly CodeSmellDetector _codeSmellDetector = new(logger);
    private readonly ComplexityAnalyzer _complexityAnalyzer = new(logger);
    private readonly PerformanceAnalyzer _performanceAnalyzer = new(logger);

    /// <inheritdoc/>
    public string Language => "csharp";

    /// <summary>
    /// Gets the programming language enum value
    /// </summary>
    public static ProgrammingLanguage LanguageEnum => ProgrammingLanguage.CSharp;

    /// <inheritdoc/>
    public Task<CodeAnalysisResult> AnalyzeAsync(string content, Dictionary<string, string>? options = null)
    {
        return AnalyzeInternalAsync(content, options)
            .ContinueWith(task =>
            {
                if (task.Result is { IsSuccess: true, Value: var success })
                {
                    return success;
                }

                var error = task.Result.Error;
                _logger.LogError(error, "Error analyzing C# code");

                return new CodeAnalysisResult
                {
                    FilePath = "memory",
                    ErrorMessage = $"Error analyzing C# code: {error.Message}",
                    Language = LanguageEnum,
                    IsSuccessful = false,
                    Errors = { $"Error analyzing C# code: {error.Message}" }
                };
            });
    }

    /// <summary>
    /// Internal implementation of the analysis that returns a Result
    /// </summary>
    private Task<Result<CodeAnalysisResult, Exception>> AnalyzeInternalAsync(
        string content, Dictionary<string, string>? options = null)
    {
        return ResultExtensions.TryAsync(async () =>
        {
            _logger.LogInformation("Analyzing C# code of length {Length}", content?.Length ?? 0);

            // Add a small delay to ensure proper async/await behavior
            await Task.Yield();

            var result = new CodeAnalysisResult
            {
                FilePath = "memory",
                ErrorMessage = string.Empty,
                Language = LanguageEnum,
                AnalyzedAt = DateTime.UtcNow,
                IsSuccessful = true
            };

            // Handle null content
            if (content == null)
            {
                result.ErrorMessage = "Content is null";
                result.IsSuccessful = false;
                result.Errors.Add("Content is null");
                return result;
            }

            // Parse options
            var includeMetrics = ParseOption(options, "IncludeMetrics", true);
            var includeStructures = ParseOption(options, "IncludeStructures", true);
            var includeIssues = ParseOption(options, "IncludeIssues", true);
            var analyzePerformance = ParseOption(options, "AnalyzePerformance", true);
            var analyzeComplexity = ParseOption(options, "AnalyzeComplexity", true);
            var analyzeMaintainability = ParseOption(options, "AnalyzeMaintainability", true);
            var analyzeSecurity = ParseOption(options, "AnalyzeSecurity", true);
            var analyzeStyle = ParseOption(options, "AnalyzeStyle", true);

            // Move CPU-intensive operations to a background thread
            await Task.Run(() =>
            {
                // Extract structures (classes, methods, etc.)
                if (includeStructures)
                {
                    result.Structures.AddRange(ExtractStructures(content));
                }

                // Calculate metrics
                if (includeMetrics)
                {
                    result.Metrics.AddRange(CalculateMetrics(content, result.Structures, analyzeComplexity, analyzeMaintainability));
                }

                // Detect issues
                if (includeIssues)
                {
                    // Detect code smells
                    if (analyzeMaintainability || analyzeStyle)
                    {
                        result.Issues.AddRange(_codeSmellDetector.DetectCodeSmells(content, "C#"));
                    }

                    // Detect complexity issues
                    if (analyzeComplexity)
                    {
                        result.Issues.AddRange(_complexityAnalyzer.DetectComplexityIssues(content, "C#", result.Structures));
                    }

                    // Detect performance issues
                    if (analyzePerformance)
                    {
                        result.Issues.AddRange(_performanceAnalyzer.DetectPerformanceIssues(content, "C#"));
                    }

                    // Detect security issues
                    if (analyzeSecurity)
                    {
                        result.Issues.AddRange(DetectSecurityIssues(content));
                    }
                }
            });

            _logger.LogInformation("Completed analysis of C# code. Found {IssueCount} issues, {MetricCount} metrics, {StructureCount} structures",
                result.Issues.Count, result.Metrics.Count, result.Structures.Count);

            return result;
        });
    }

    /// <summary>
    /// Extracts code structures from C# code
    /// </summary>
    public List<CodeStructure> ExtractStructures(string content)
    {
        if (string.IsNullOrWhiteSpace(content))
        {
            return [];
        }

        var structures = new List<CodeStructure>();

        try
        {
            // Extract namespaces
            var namespaceMatches = RegexPatterns.CSharpNamespace().Matches(content);
            foreach (Match match in namespaceMatches)
            {
                var namespaceName = match.Groups[1].Value;
                var startPos = match.Index;
                var endPos = FindMatchingBrace(content, match.Index + match.Length - 1);
                var namespaceContent = content[startPos..(endPos + 1)];

                structures.Add(new CodeStructure
                {
                    Type = StructureType.Namespace,
                    Name = namespaceName,
                    Location = new CodeLocation
                    {
                        StartLine = GetLineNumber(content, startPos),
                        EndLine = GetLineNumber(content, endPos),
                        Namespace = namespaceName
                    },
                    Size = namespaceContent.Length
                });
            }

            // Extract classes
            var classMatches = RegexPatterns.CSharpClass().Matches(content);
            foreach (Match match in classMatches)
            {
                var className = match.Groups[3].Value;
                var startPos = match.Index;
                var openBracePos = content.IndexOf('{', startPos);
                if (openBracePos == -1) continue;

                var endPos = FindMatchingBrace(content, openBracePos);
                if (endPos == -1) continue;

                var classContent = content[startPos..(endPos + 1)];
                var lineNumber = GetLineNumber(content, startPos);
                var endLineNumber = GetLineNumber(content, endPos);
                var containingNamespace = GetNamespaceForPosition(structures, startPos, content);

                structures.Add(new CodeStructure
                {
                    Type = StructureType.Class,
                    Name = className,
                    Location = new CodeLocation
                    {
                        StartLine = lineNumber,
                        EndLine = endLineNumber,
                        Namespace = containingNamespace,
                        ClassName = className
                    },
                    ParentName = containingNamespace,
                    Size = classContent.Length
                });
            }

            // Extract interfaces
            var interfaceMatches = RegexPatterns.CSharpInterface().Matches(content);
            foreach (Match match in interfaceMatches)
            {
                var interfaceName = match.Groups[2].Value;
                var startPos = match.Index;
                var openBracePos = content.IndexOf('{', startPos);
                if (openBracePos == -1) continue;

                var endPos = FindMatchingBrace(content, openBracePos);
                if (endPos == -1) continue;

                var interfaceContent = content[startPos..(endPos + 1)];
                var lineNumber = GetLineNumber(content, startPos);
                var endLineNumber = GetLineNumber(content, endPos);
                var containingNamespace = GetNamespaceForPosition(structures, startPos, content);

                structures.Add(new CodeStructure
                {
                    Type = StructureType.Interface,
                    Name = interfaceName,
                    Location = new CodeLocation
                    {
                        StartLine = lineNumber,
                        EndLine = endLineNumber,
                        Namespace = containingNamespace,
                        ClassName = interfaceName
                    },
                    ParentName = containingNamespace,
                    Size = interfaceContent.Length
                });
            }

            // Extract methods
            var methodMatches = RegexPatterns.CSharpMethod().Matches(content);
            foreach (Match match in methodMatches)
            {
                var methodName = match.Groups[4].Value;
                var returnType = match.Groups[3].Value;
                var startPos = match.Index;
                var openBracePos = content.IndexOf('{', startPos);
                if (openBracePos == -1) continue;

                var endPos = FindMatchingBrace(content, openBracePos);
                if (endPos == -1) continue;

                var methodContent = content[startPos..(endPos + 1)];
                var lineNumber = GetLineNumber(content, startPos);
                var endLineNumber = GetLineNumber(content, endPos);
                var containingClass = GetClassForPosition(structures, startPos, content);
                var containingNamespace = GetNamespaceForPosition(structures, startPos, content);

                structures.Add(new CodeStructure
                {
                    Type = StructureType.Method,
                    Name = methodName,
                    Location = new CodeLocation
                    {
                        StartLine = lineNumber,
                        EndLine = endLineNumber,
                        Namespace = containingNamespace,
                        ClassName = containingClass,
                        MethodName = methodName
                    },
                    ParentName = containingClass,
                    Size = methodContent.Length,
                    Properties = new Dictionary<string, string>
                    {
                        { "ReturnType", returnType }
                    }
                });
            }

            // Extract properties
            var propertyMatches = RegexPatterns.CSharpProperty().Matches(content);
            foreach (Match match in propertyMatches)
            {
                var propertyName = match.Groups[4].Value;
                var propertyType = match.Groups[3].Value;
                var startPos = match.Index;
                var openBracePos = content.IndexOf('{', startPos);
                if (openBracePos == -1) continue;

                var endPos = FindMatchingBrace(content, openBracePos);
                if (endPos == -1) continue;

                var propertyContent = content[startPos..(endPos + 1)];
                var lineNumber = GetLineNumber(content, startPos);
                var endLineNumber = GetLineNumber(content, endPos);
                var containingClass = GetClassForPosition(structures, startPos, content);
                var containingNamespace = GetNamespaceForPosition(structures, startPos, content);

                structures.Add(new CodeStructure
                {
                    Type = StructureType.Property,
                    Name = propertyName,
                    Location = new CodeLocation
                    {
                        StartLine = lineNumber,
                        EndLine = endLineNumber,
                        Namespace = containingNamespace,
                        ClassName = containingClass
                    },
                    ParentName = containingClass,
                    Size = propertyContent.Length,
                    Properties = new Dictionary<string, string>
                    {
                        { "PropertyType", propertyType }
                    }
                });
            }

            // Calculate sizes
            foreach (var structure in structures)
            {
// Size already set
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error extracting structures from C# code");
        }

        return structures;
    }

    /// <summary>
    /// Calculates code metrics for C# code
    /// </summary>
    public List<CodeMetric> CalculateMetrics(string content, List<CodeStructure> structures, bool analyzeComplexity, bool analyzeMaintainability)
    {
        var metrics = new List<CodeMetric>();

        try
        {
            // Add basic metrics
            metrics.Add(new CodeMetric
            {
                Type = MetricType.Size,
                Value = content.Count(c => c == '\n') + 1,
                Name = "Lines of Code"
            });

            metrics.Add(new CodeMetric
            {
                Type = MetricType.Size,
                Value = content.Length,
                Name = "Character Count"
            });

            // Add structure counts
            metrics.Add(new CodeMetric
            {
                Type = MetricType.Size,
                Value = structures.Count(s => s.Type == StructureType.Class),
                Name = "Class Count"
            });

            metrics.Add(new CodeMetric
            {
                Type = MetricType.Size,
                Value = structures.Count(s => s.Type == StructureType.Method),
                Name = "Method Count"
            });

            metrics.Add(new CodeMetric
            {
                Type = MetricType.Size,
                Value = structures.Count(s => s.Type == StructureType.Interface),
                Name = "Interface Count"
            });

            // Calculate Halstead metrics if complexity analysis is enabled
            if (analyzeComplexity)
            {
                // Count unique operators
                var operatorMatches = RegexPatterns.Operators().Matches(content);
                var uniqueOperators = new HashSet<string>();
                foreach (Match match in operatorMatches)
                {
                    uniqueOperators.Add(match.Value);
                }

                // Count unique operands (identifiers and literals)
                var operandMatches = RegexPatterns.Operands().Matches(content);
                var uniqueOperands = new HashSet<string>();
                foreach (Match match in operandMatches)
                {
                    uniqueOperands.Add(match.Value);
                }

                var n1 = uniqueOperators.Count;
                var n2 = uniqueOperands.Count;
                var N1 = operatorMatches.Count;
                var N2 = operandMatches.Count;

                // Calculate Halstead metrics
                var programVocabulary = n1 + n2;
                var programLength = N1 + N2;
                var calculatedProgramLength = n1 * Math.Log2(n1) + n2 * Math.Log2(n2);
                var volume = programLength * Math.Log2(programVocabulary);
                var difficulty = (n1 / 2.0) * (N2 / (double)n2);
                var effort = difficulty * volume;
                var timeToImplement = effort / 18.0; // Time in seconds
                var deliveredBugs = Math.Pow(effort, 2.0/3.0) / 3000.0;

                metrics.Add(new CodeMetric
                {
                    Type = MetricType.Complexity,
                    Value = programVocabulary,
                    Name = "Halstead Vocabulary"
                });

                metrics.Add(new CodeMetric
                {
                    Type = MetricType.Complexity,
                    Value = programLength,
                    Name = "Halstead Length"
                });

                metrics.Add(new CodeMetric
                {
                    Type = MetricType.Complexity,
                    Value = volume,
                    Name = "Halstead Volume"
                });

                metrics.Add(new CodeMetric
                {
                    Type = MetricType.Complexity,
                    Value = difficulty,
                    Name = "Halstead Difficulty"
                });

                metrics.Add(new CodeMetric
                {
                    Type = MetricType.Complexity,
                    Value = effort,
                    Name = "Halstead Effort"
                });

                metrics.Add(new CodeMetric
                {
                    Type = MetricType.Complexity,
                    Value = timeToImplement,
                    Name = "Halstead Time (seconds)"
                });

                metrics.Add(new CodeMetric
                {
                    Type = MetricType.Complexity,
                    Value = deliveredBugs,
                    Name = "Halstead Bugs"
                });
            }

            // Calculate cyclomatic complexity for each method
            if (analyzeComplexity)
            {
                var methodStructures = structures.Where(s => s.Type == StructureType.Method).ToList();
                foreach (var method in methodStructures)
                {
                    // Get method content from the file
                    var methodStartLine = method.Location.StartLine;
                    var methodEndLine = method.Location.EndLine;
                    var methodContent = string.Join("\n", content.Split('\n').Skip(methodStartLine).Take(methodEndLine - methodStartLine + 1));
                    var complexity = CalculateCyclomaticComplexity(methodContent);

                    metrics.Add(new CodeMetric
                    {
                        Type = MetricType.Complexity,
                        Value = complexity,
                        Name = $"Cyclomatic Complexity: {method.Name}",
                        Target = method.Name
                    });
                }

                // Calculate average cyclomatic complexity
                var complexityMetrics = metrics.Where(m => m.Name.StartsWith("Cyclomatic Complexity:")).ToList();
                if (complexityMetrics.Count > 0)
                {
                    var avgComplexity = complexityMetrics.Average(m => m.Value);
                    metrics.Add(new CodeMetric
                    {
                        Type = MetricType.Complexity,
                        Value = avgComplexity,
                        Name = "Average Cyclomatic Complexity"
                    });
                }
            }

            // Calculate maintainability index if enabled
            if (analyzeMaintainability && analyzeComplexity)
            {
                // Get Halstead volume
                var halsteadVolume = metrics.FirstOrDefault(m => m.Name == "Halstead Volume")?.Value ?? 0;

                // Get average cyclomatic complexity
                var avgComplexity = metrics.FirstOrDefault(m => m.Name == "Average Cyclomatic Complexity")?.Value ?? 0;

                // Get lines of code
                var linesOfCode = metrics.FirstOrDefault(m => m.Name == "Lines of Code")?.Value ?? 0;

                // Calculate maintainability index
                // MI = 171 - 5.2 * ln(HV) - 0.23 * CC - 16.2 * ln(LOC)
                var maintainabilityIndex = 171 - 5.2 * Math.Log(Math.Max(1, halsteadVolume)) - 0.23 * avgComplexity - 16.2 * Math.Log(Math.Max(1, linesOfCode));
                maintainabilityIndex = Math.Max(0, Math.Min(100, maintainabilityIndex));

                metrics.Add(new CodeMetric
                {
                    Type = MetricType.Maintainability,
                    Value = maintainabilityIndex,
                    Name = "Maintainability Index"
                });
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error calculating metrics for C# code");
        }

        return metrics;
    }

    /// <summary>
    /// Detects security issues in C# code
    /// </summary>
    public List<CodeIssue> DetectSecurityIssues(string content)
    {
        var issues = new List<CodeIssue>();

        try
        {
            // Detect SQL injection vulnerabilities
            var sqlInjectionMatches = RegexPatterns.SqlInjection().Matches(content);
            foreach (Match match in sqlInjectionMatches)
            {
                var lineNumber = GetLineNumber(content, match.Index);
                issues.Add(new CodeIssue
                {
                    Type = CodeIssueType.Security,
                    Severity = TarsEngine.Models.IssueSeverity.Critical,
                    Title = "Potential SQL injection vulnerability detected",
                    Description = "SQL injection vulnerability detected",
                    Location = new CodeLocation { StartLine = lineNumber },
                    CodeSnippet = match.Value,
                    SuggestedFix = "Use parameterized queries or an ORM instead of string concatenation"
                });
            }

            // Detect XSS vulnerabilities
            var xssMatches = RegexPatterns.XssVulnerability().Matches(content);
            foreach (Match match in xssMatches)
            {
                var lineNumber = GetLineNumber(content, match.Index);
                issues.Add(new CodeIssue
                {
                    Type = CodeIssueType.Security,
                    Severity = TarsEngine.Models.IssueSeverity.Critical,
                    Title = "Potential XSS vulnerability detected",
                    Description = "XSS vulnerability detected",
                    Location = new CodeLocation { StartLine = lineNumber },
                    CodeSnippet = match.Value,
                    SuggestedFix = "Use HTML encoding or a templating engine that automatically escapes output"
                });
            }

            // Detect hardcoded credentials
            var credentialsMatches = RegexPatterns.HardcodedCredentials().Matches(content);
            foreach (Match match in credentialsMatches)
            {
                var lineNumber = GetLineNumber(content, match.Index);
                issues.Add(new CodeIssue
                {
                    Type = CodeIssueType.Security,
                    Severity = TarsEngine.Models.IssueSeverity.Critical,
                    Title = "Hardcoded credentials detected",
                    Description = "Hardcoded credentials detected",
                    Location = new CodeLocation { StartLine = lineNumber },
                    CodeSnippet = match.Value,
                    SuggestedFix = "Store credentials in a secure configuration system or environment variables"
                });
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error detecting security issues in C# code");
        }

        return issues;
    }

    /// <summary>
    /// Calculates the cyclomatic complexity of a method
    /// </summary>
    private static int CalculateCyclomaticComplexity(string methodContent)
    {
        // Start with 1 (base complexity)
        var complexity = 1;

        // Count decision points
        complexity += Regex.Matches(methodContent, @"\bif\b").Count;
        complexity += Regex.Matches(methodContent, @"\belse\s+if\b").Count;
        complexity += Regex.Matches(methodContent, @"\bwhile\b").Count;
        complexity += Regex.Matches(methodContent, @"\bfor\b").Count;
        complexity += Regex.Matches(methodContent, @"\bforeach\b").Count;
        complexity += Regex.Matches(methodContent, @"\bcase\b").Count;
        complexity += Regex.Matches(methodContent, @"\bcatch\b").Count;
        complexity += Regex.Matches(methodContent, @"\b\|\|\b").Count;
        complexity += Regex.Matches(methodContent, @"\b&&\b").Count;
        complexity += Regex.Matches(methodContent, @"\?\s*[^:]+\s*:").Count; // Ternary operators

        return complexity;
    }

    /// <summary>
    /// Finds the position of the matching closing brace
    /// </summary>
    private static int FindMatchingBrace(string content, int openBracePos)
    {
        var braceCount = 1;
        for (var i = openBracePos + 1; i < content.Length; i++)
        {
            if (content[i] == '{')
            {
                braceCount++;
            }
            else if (content[i] == '}')
            {
                braceCount--;
                if (braceCount == 0)
                {
                    return i;
                }
            }
        }
        return -1;
    }

    /// <summary>
    /// Gets the line number for a position in the content
    /// </summary>
    private static int GetLineNumber(string content, int position)
    {
        // Count newlines before the position
        return content[..position].Count(c => c == '\n');
    }

    /// <summary>
    /// Gets the namespace for a position in the content
    /// </summary>
    private static string GetNamespaceForPosition(List<CodeStructure> structures, int position, string content)
    {
        var lineNumber = GetLineNumber(content, position);
        var containingNamespace = structures
            .Where(s => s.Type == StructureType.Namespace)
            .FirstOrDefault(s => lineNumber >= s.Location.StartLine && lineNumber <= s.Location.EndLine);

        return containingNamespace?.Name ?? string.Empty;
    }

    /// <summary>
    /// Gets the class for a position in the content
    /// </summary>
    private static string GetClassForPosition(List<CodeStructure> structures, int position, string content)
    {
        var lineNumber = GetLineNumber(content, position);
        var containingClass = structures
            .Where(s => s.Type == StructureType.Class)
            .FirstOrDefault(s => lineNumber >= s.Location.StartLine && lineNumber <= s.Location.EndLine);

        return containingClass?.Name ?? string.Empty;
    }

    /// <summary>
    /// Gets the available options for the analyzer
    /// </summary>
    public async Task<Dictionary<string, string>> GetAvailableOptionsAsync()
    {
        try
        {
            // Since this is CPU-bound work, we'll move it to a background thread
            return await Task.Run(() => new Dictionary<string, string>
            {
                { "IncludeMetrics", "true/false - Include code metrics in the analysis" },
                { "IncludeStructures", "true/false - Include code structures in the analysis" },
                { "IncludeIssues", "true/false - Include code issues in the analysis" },
                { "AnalyzePerformance", "true/false - Analyze performance issues" },
                { "AnalyzeComplexity", "true/false - Analyze code complexity" },
                { "AnalyzeMaintainability", "true/false - Analyze code maintainability" },
                { "AnalyzeSecurity", "true/false - Analyze security issues" },
                { "AnalyzeStyle", "true/false - Analyze code style issues" }
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting available options for C# analyzer");
            return new Dictionary<string, string>();
        }
    }

    /// <summary>
    /// Gets the language-specific issue types
    /// </summary>
    public Task<Dictionary<CodeIssueType, string>> GetLanguageSpecificIssueTypesAsync()
    {
        return Task.FromResult(new Dictionary<CodeIssueType, string>
        {
            { CodeIssueType.CodeSmell, "Code smell in C# code" },
            { CodeIssueType.Security, "Security vulnerability in C# code" },
            { CodeIssueType.Performance, "Performance issue in C# code" },
            { CodeIssueType.Complexity, "Complexity issue in C# code" },
            { CodeIssueType.Style, "Style issue in C# code" }
        });
    }

    /// <summary>
    /// Gets the language-specific metric types
    /// </summary>
    public Task<Dictionary<MetricType, string>> GetLanguageSpecificMetricTypesAsync()
    {
        // We can't have duplicate keys in a dictionary, so we'll just return the unique metric types
        return Task.FromResult(new Dictionary<MetricType, string>
        {
            { MetricType.Size, "Size metrics for C# code" },
            { MetricType.Complexity, "Complexity metrics for C# code" },
            { MetricType.Maintainability, "Maintainability metrics for C# code" },
            { MetricType.Performance, "Performance metrics for C# code" }
        });
    }

    /// <summary>
    /// Parses a boolean option from the options dictionary
    /// </summary>
    private static bool ParseOption(Dictionary<string, string>? options, string key, bool defaultValue)
    {
        if (options == null || !options.TryGetValue(key, out var value))
        {
            return defaultValue;
        }

        return value.Equals("true", StringComparison.OrdinalIgnoreCase);
    }
}
