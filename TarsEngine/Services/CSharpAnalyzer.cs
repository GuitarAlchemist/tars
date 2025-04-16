using System.Text.RegularExpressions;
using Microsoft.Extensions.Logging;
using TarsEngine.Models;
using TarsEngine.Services.Interfaces;

namespace TarsEngine.Services;

/// <summary>
/// Analyzer for C# code
/// </summary>
public class CSharpAnalyzer : ILanguageAnalyzer
{
    private readonly ILogger _logger;
    private readonly CodeSmellDetector _codeSmellDetector;
    private readonly ComplexityAnalyzer _complexityAnalyzer;
    private readonly PerformanceAnalyzer _performanceAnalyzer;

    /// <summary>
    /// Initializes a new instance of the <see cref="CSharpAnalyzer"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    public CSharpAnalyzer(ILogger logger)
    {
        _logger = logger;
        _codeSmellDetector = new CodeSmellDetector(logger);
        _complexityAnalyzer = new ComplexityAnalyzer(logger);
        _performanceAnalyzer = new PerformanceAnalyzer(logger);
    }

    /// <inheritdoc/>
    public string Language => "csharp";

    /// <summary>
    /// Gets the programming language enum value
    /// </summary>
    public static ProgrammingLanguage LanguageEnum => ProgrammingLanguage.CSharp;

    /// <inheritdoc/>
    public async Task<CodeAnalysisResult> AnalyzeAsync(string content, Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Analyzing C# code of length {Length}", content?.Length ?? 0);

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

            // Add a small delay to ensure the async method actually awaits something
            await Task.Delay(1);

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

            _logger.LogInformation("Completed analysis of C# code. Found {IssueCount} issues, {MetricCount} metrics, {StructureCount} structures",
                result.Issues.Count, result.Metrics.Count, result.Structures.Count);

            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing C# code");
            return new CodeAnalysisResult
            {
                FilePath = "memory",
                ErrorMessage = $"Error analyzing C# code: {ex.Message}",
                Language = LanguageEnum,
                IsSuccessful = false,
                Errors = { $"Error analyzing C# code: {ex.Message}" }
            };
        }
    }

    /// <inheritdoc/>
    public Task<Dictionary<string, string>> GetAvailableOptionsAsync()
    {
        var options = new Dictionary<string, string>
        {
            { "AnalyzeNullableReferences", "Whether to analyze nullable reference types (true/false)" },
            { "AnalyzeAsyncAwait", "Whether to analyze async/await patterns (true/false)" },
            { "AnalyzeLINQ", "Whether to analyze LINQ queries (true/false)" },
            { "AnalyzeDisposable", "Whether to analyze IDisposable implementations (true/false)" },
            { "AnalyzeExceptions", "Whether to analyze exception handling (true/false)" },
            { "AnalyzeGenerics", "Whether to analyze generic type usage (true/false)" },
            { "AnalyzeReflection", "Whether to analyze reflection usage (true/false)" },
            { "AnalyzeThreading", "Whether to analyze threading and concurrency (true/false)" },
            { "AnalyzeUnsafe", "Whether to analyze unsafe code (true/false)" },
            { "AnalyzeInterop", "Whether to analyze interop code (true/false)" }
        };
        return Task.FromResult(options);
    }

    /// <inheritdoc/>
    public Task<Dictionary<CodeIssueType, string>> GetLanguageSpecificIssueTypesAsync()
    {
        var issueTypes = new Dictionary<CodeIssueType, string>
        {
            { CodeIssueType.CodeSmell, "C#-specific code smells like unused using directives, redundant casts, etc." },
            { CodeIssueType.Performance, "C#-specific performance issues like inefficient LINQ queries, boxing/unboxing, etc." },
            { CodeIssueType.Maintainability, "C#-specific maintainability issues like complex lambda expressions, nested using statements, etc." },
            { CodeIssueType.Security, "C#-specific security issues like SQL injection, XSS vulnerabilities, etc." },
            { CodeIssueType.Design, "C#-specific design issues like improper interface implementations, inheritance issues, etc." }
        };
        return Task.FromResult(issueTypes);
    }

    /// <inheritdoc/>
    public Task<Dictionary<MetricType, string>> GetLanguageSpecificMetricTypesAsync()
    {
        var metricTypes = new Dictionary<MetricType, string>
        {
            { MetricType.Complexity, "C#-specific complexity metrics like cyclomatic complexity, cognitive complexity, etc." },
            { MetricType.Coupling, "C#-specific coupling metrics like afferent coupling, efferent coupling, etc." },
            { MetricType.Cohesion, "C#-specific cohesion metrics like lack of cohesion in methods, etc." },
            { MetricType.Inheritance, "C#-specific inheritance metrics like depth of inheritance tree, number of children, etc." }
        };
        return Task.FromResult(metricTypes);
    }

    private List<CodeStructure> ExtractStructures(string content)
    {
        var structures = new List<CodeStructure>();

        try
        {
            // Extract namespaces
            var namespaceRegex = new Regex(@"namespace\s+([a-zA-Z0-9_.]+)\s*{", RegexOptions.Compiled);
            var namespaceMatches = namespaceRegex.Matches(content);
            foreach (Match match in namespaceMatches)
            {
                if (match.Groups.Count > 1)
                {
                    var namespaceName = match.Groups[1].Value;
                    structures.Add(new CodeStructure
                    {
                        Type = StructureType.Namespace,
                        Name = namespaceName,
                        Location = new CodeLocation
                        {
                            StartLine = GetLineNumber(content, match.Index),
                            Namespace = namespaceName
                        }
                    });
                }
            }

            // Extract classes
            var classRegex = new Regex(@"(public|private|protected|internal)?\s*(static|abstract|sealed)?\s*class\s+([a-zA-Z0-9_]+)(?:<[^>]+>)?(?:\s*:\s*[^{]+)?", RegexOptions.Compiled);
            var classMatches = classRegex.Matches(content);
            foreach (Match match in classMatches)
            {
                if (match.Groups.Count > 3)
                {
                    var className = match.Groups[3].Value;
                    var namespaceName = GetNamespaceForPosition(structures, match.Index, content);
                    structures.Add(new CodeStructure
                    {
                        Type = StructureType.Class,
                        Name = className,
                        Location = new CodeLocation
                        {
                            StartLine = GetLineNumber(content, match.Index),
                            Namespace = namespaceName,
                            ClassName = className
                        }
                    });
                }
            }

            // Extract interfaces
            var interfaceRegex = new Regex(@"(public|private|protected|internal)?\s*interface\s+([a-zA-Z0-9_]+)(?:<[^>]+>)?(?:\s*:\s*[^{]+)?", RegexOptions.Compiled);
            var interfaceMatches = interfaceRegex.Matches(content);
            foreach (Match match in interfaceMatches)
            {
                if (match.Groups.Count > 2)
                {
                    var interfaceName = match.Groups[2].Value;
                    var namespaceName = GetNamespaceForPosition(structures, match.Index, content);
                    structures.Add(new CodeStructure
                    {
                        Type = StructureType.Interface,
                        Name = interfaceName,
                        Location = new CodeLocation
                        {
                            StartLine = GetLineNumber(content, match.Index),
                            Namespace = namespaceName,
                            ClassName = interfaceName
                        }
                    });
                }
            }

            // Extract methods
            var methodRegex = new Regex(@"(public|private|protected|internal)?\s*(static|virtual|abstract|override|async)?\s*([a-zA-Z0-9_<>]+)\s+([a-zA-Z0-9_]+)\s*\(", RegexOptions.Compiled);
            var methodMatches = methodRegex.Matches(content);
            foreach (Match match in methodMatches)
            {
                if (match.Groups.Count > 4)
                {
                    var methodName = match.Groups[4].Value;
                    var className = GetClassForPosition(structures, match.Index, content);
                    var namespaceName = GetNamespaceForPosition(structures, match.Index, content);

                    // Skip if this is a constructor (method name same as class name)
                    if (methodName == className)
                    {
                        continue;
                    }

                    // Skip if this is a property accessor (get or set)
                    if (methodName == "get" || methodName == "set")
                    {
                        continue;
                    }

                    structures.Add(new CodeStructure
                    {
                        Type = StructureType.Method,
                        Name = methodName,
                        ParentName = className,
                        Location = new CodeLocation
                        {
                            StartLine = GetLineNumber(content, match.Index),
                            Namespace = namespaceName,
                            ClassName = className,
                            MethodName = methodName
                        }
                    });
                }
            }

            // Extract properties
            var propertyRegex = new Regex(@"(public|private|protected|internal)?\s*(static|virtual|abstract|override)?\s*([a-zA-Z0-9_<>]+)\s+([a-zA-Z0-9_]+)\s*{\s*(get|set)?", RegexOptions.Compiled);
            var propertyMatches = propertyRegex.Matches(content);
            foreach (Match match in propertyMatches)
            {
                if (match.Groups.Count > 4)
                {
                    var propertyName = match.Groups[4].Value;
                    var className = GetClassForPosition(structures, match.Index, content);
                    var namespaceName = GetNamespaceForPosition(structures, match.Index, content);

                    structures.Add(new CodeStructure
                    {
                        Type = StructureType.Property,
                        Name = propertyName,
                        ParentName = className,
                        Location = new CodeLocation
                        {
                            StartLine = GetLineNumber(content, match.Index),
                            Namespace = namespaceName,
                            ClassName = className
                        }
                    });
                }
            }

            // Calculate structure sizes and update end lines
            CalculateStructureSizes(structures, content);

            return structures;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error extracting structures from C# code");
            return structures;
        }
    }

    private List<CodeMetric> CalculateMetrics(string content, List<CodeStructure> structures, bool analyzeComplexity, bool analyzeMaintainability)
    {
        var metrics = new List<CodeMetric>();

        try
        {
            // File-level metrics
            metrics.Add(new CodeMetric
            {
                Name = "Lines of Code",
                Value = content.Split('\n').Length,
                Type = MetricType.Size,
                Scope = MetricScope.File
            });

            // Calculate metrics for each structure
            foreach (var structure in structures)
            {
                switch (structure.Type)
                {
                    case StructureType.Class:
                        // Class size
                        metrics.Add(new CodeMetric
                        {
                            Name = "Class Size",
                            Value = structure.Size,
                            Type = MetricType.Size,
                            Scope = MetricScope.Class,
                            Target = structure.Name,
                            Location = structure.Location
                        });

                        // Class complexity
                        if (analyzeComplexity)
                        {
                            var classComplexity = _complexityAnalyzer.CalculateClassComplexity(content, structure);
                            metrics.Add(new CodeMetric
                            {
                                Name = "Class Complexity",
                                Value = classComplexity,
                                Type = MetricType.Complexity,
                                Scope = MetricScope.Class,
                                Target = structure.Name,
                                Location = structure.Location
                            });
                        }

                        // Class maintainability
                        if (analyzeMaintainability)
                        {
                            var maintainabilityIndex = CalculateMaintainabilityIndex(content, structure);
                            metrics.Add(new CodeMetric
                            {
                                Name = "Maintainability Index",
                                Value = maintainabilityIndex,
                                Type = MetricType.Maintainability,
                                Scope = MetricScope.Class,
                                Target = structure.Name,
                                Location = structure.Location
                            });
                        }
                        break;

                    case StructureType.Method:
                        // Method size
                        metrics.Add(new CodeMetric
                        {
                            Name = "Method Size",
                            Value = structure.Size,
                            Type = MetricType.Size,
                            Scope = MetricScope.Method,
                            Target = structure.Name,
                            Location = structure.Location
                        });

                        // Method complexity
                        if (analyzeComplexity)
                        {
                            var methodComplexity = _complexityAnalyzer.CalculateMethodComplexity(content, structure);
                            metrics.Add(new CodeMetric
                            {
                                Name = "Cyclomatic Complexity",
                                Value = methodComplexity,
                                Type = MetricType.Complexity,
                                Scope = MetricScope.Method,
                                Target = structure.Name,
                                Location = structure.Location
                            });

                            var cognitiveComplexity = _complexityAnalyzer.CalculateCognitiveComplexity(content, structure);
                            metrics.Add(new CodeMetric
                            {
                                Name = "Cognitive Complexity",
                                Value = cognitiveComplexity,
                                Type = MetricType.Complexity,
                                Scope = MetricScope.Method,
                                Target = structure.Name,
                                Location = structure.Location
                            });
                        }
                        break;
                }
            }

            return metrics;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error calculating metrics for C# code");
            return metrics;
        }
    }

    private List<CodeIssue> DetectSecurityIssues(string content)
    {
        var issues = new List<CodeIssue>();

        try
        {
            // Detect SQL injection vulnerabilities
            var sqlInjectionRegex = new Regex(@"SqlCommand\s*\(\s*[""'].*?\+\s*[^""']+\s*\+", RegexOptions.Compiled);
            var sqlInjectionMatches = sqlInjectionRegex.Matches(content);
            foreach (Match match in sqlInjectionMatches)
            {
                issues.Add(new CodeIssue
                {
                    Type = CodeIssueType.Vulnerability,
                    Severity = TarsEngine.Models.IssueSeverity.Critical,
                    Title = "Potential SQL Injection",
                    Description = "String concatenation in SQL commands can lead to SQL injection vulnerabilities.",
                    Location = new CodeLocation
                    {
                        StartLine = GetLineNumber(content, match.Index)
                    },
                    CodeSnippet = GetCodeSnippet(content, match.Index),
                    SuggestedFix = "Use parameterized queries instead of string concatenation.",
                    ImpactScore = 0.9,
                    FixDifficultyScore = 0.3,
                    Tags = { "security", "sql-injection" }
                });
            }

            // Detect XSS vulnerabilities
            var xssRegex = new Regex(@"Response\.Write\s*\(\s*[^""']*\s*\)", RegexOptions.Compiled);
            var xssMatches = xssRegex.Matches(content);
            foreach (Match match in xssMatches)
            {
                issues.Add(new CodeIssue
                {
                    Type = CodeIssueType.Vulnerability,
                    Severity = TarsEngine.Models.IssueSeverity.Critical,
                    Title = "Potential Cross-Site Scripting (XSS)",
                    Description = "Directly writing user input to the response can lead to XSS vulnerabilities.",
                    Location = new CodeLocation
                    {
                        StartLine = GetLineNumber(content, match.Index)
                    },
                    CodeSnippet = GetCodeSnippet(content, match.Index),
                    SuggestedFix = "Encode user input before writing to the response using HttpUtility.HtmlEncode().",
                    ImpactScore = 0.9,
                    FixDifficultyScore = 0.3,
                    Tags = { "security", "xss" }
                });
            }

            // Detect hardcoded credentials
            var credentialsRegex = new Regex(@"(password|pwd|passwd|secret|key|token|apikey)\s*=\s*[""'][^""']+[""']", RegexOptions.IgnoreCase | RegexOptions.Compiled);
            var credentialsMatches = credentialsRegex.Matches(content);
            foreach (Match match in credentialsMatches)
            {
                issues.Add(new CodeIssue
                {
                    Type = CodeIssueType.SecurityHotspot,
                    Severity = TarsEngine.Models.IssueSeverity.Critical,
                    Title = "Hardcoded Credentials",
                    Description = "Hardcoded credentials can lead to security vulnerabilities.",
                    Location = new CodeLocation
                    {
                        StartLine = GetLineNumber(content, match.Index)
                    },
                    CodeSnippet = GetCodeSnippet(content, match.Index),
                    SuggestedFix = "Store credentials in a secure configuration system or use a secret manager.",
                    ImpactScore = 0.9,
                    FixDifficultyScore = 0.5,
                    Tags = { "security", "credentials" }
                });
            }

            return issues;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error detecting security issues in C# code");
            return issues;
        }
    }

    private double CalculateMaintainabilityIndex(string content, CodeStructure structure)
    {
        try
        {
            // Extract the structure's content
            var structureContent = ExtractStructureContent(content, structure);

            // Calculate Halstead volume (simplified)
            var halsteadVolume = CalculateHalsteadVolume(structureContent);

            // Calculate cyclomatic complexity
            var cyclomaticComplexity = _complexityAnalyzer.CalculateClassComplexity(content, structure);

            // Calculate lines of code
            var linesOfCode = structure.Size;

            // Calculate maintainability index using the formula:
            // MI = 171 - 5.2 * ln(HV) - 0.23 * CC - 16.2 * ln(LOC)
            var maintainabilityIndex = 171 - 5.2 * Math.Log(halsteadVolume) - 0.23 * cyclomaticComplexity - 16.2 * Math.Log(linesOfCode);

            // Normalize to 0-100 scale
            maintainabilityIndex = Math.Max(0, Math.Min(100, maintainabilityIndex));

            return maintainabilityIndex;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error calculating maintainability index for structure {Name}", structure.Name);
            return 50; // Default value
        }
    }

    private double CalculateHalsteadVolume(string content)
    {
        try
        {
            // Count unique operators
            var operatorRegex = new Regex(@"[+\-*/=<>!&|^~%]|==|!=|<=|>=|&&|\|\||<<|>>|\+\+|--|->|\+=|-=|\*=|/=|%=|&=|\|=|\^=|<<=|>>=", RegexOptions.Compiled);
            var operatorMatches = operatorRegex.Matches(content);
            var uniqueOperators = new HashSet<string>();
            foreach (Match match in operatorMatches)
            {
                uniqueOperators.Add(match.Value);
            }

            // Count unique operands (identifiers and literals)
            var operandRegex = new Regex(@"\b[a-zA-Z_][a-zA-Z0-9_]*\b|""[^""]*""|'[^']*'|\d+(\.\d+)?", RegexOptions.Compiled);
            var operandMatches = operandRegex.Matches(content);
            var uniqueOperands = new HashSet<string>();
            foreach (Match match in operandMatches)
            {
                uniqueOperands.Add(match.Value);
            }

            // Calculate Halstead metrics
            var n1 = uniqueOperators.Count;
            var n2 = uniqueOperands.Count;
            var N1 = operatorMatches.Count;
            var N2 = operandMatches.Count;

            // Avoid division by zero or log of zero
            if (n1 == 0 || n2 == 0)
            {
                return 0;
            }

            // Calculate vocabulary and length
            var vocabulary = n1 + n2;
            var length = N1 + N2;

            // Calculate volume
            var volume = length * Math.Log2(vocabulary);

            return volume;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error calculating Halstead volume");
            return 0;
        }
    }

    private string ExtractStructureContent(string content, CodeStructure structure)
    {
        try
        {
            // Get the lines for the structure
            var lines = content.Split('\n');
            var startLine = structure.Location.StartLine;
            var endLine = startLine + structure.Size - 1;

            // Ensure valid line numbers
            startLine = Math.Max(0, Math.Min(startLine, lines.Length - 1));
            endLine = Math.Max(startLine, Math.Min(endLine, lines.Length - 1));

            // Extract the content
            var structureLines = lines.Skip(startLine).Take(endLine - startLine + 1);
            return string.Join("\n", structureLines);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error extracting structure content for {Name}", structure.Name);
            return string.Empty;
        }
    }

    private void CalculateStructureSizes(List<CodeStructure> structures, string content)
    {
        try
        {
            // Sort structures by start line
            var sortedStructures = structures.OrderBy(s => s.Location.StartLine).ToList();

            // Calculate sizes and end lines
            for (var i = 0; i < sortedStructures.Count; i++)
            {
                var structure = sortedStructures[i];

                // Find the next structure at the same or higher level
                var nextStructureIndex = -1;
                for (var j = i + 1; j < sortedStructures.Count; j++)
                {
                    var nextStructure = sortedStructures[j];

                    // If this is a child structure, skip it
                    if (nextStructure.ParentName == structure.Name)
                    {
                        continue;
                    }

                    // If this is at the same level or higher, use it
                    if (nextStructure.Type == structure.Type ||
                        (structure.Type == StructureType.Namespace && nextStructure.Type == StructureType.Namespace) ||
                        (structure.Type == StructureType.Class && nextStructure.Type == StructureType.Class) ||
                        (structure.Type == StructureType.Interface && nextStructure.Type == StructureType.Interface))
                    {
                        nextStructureIndex = j;
                        break;
                    }
                }

                // Calculate end line
                int endLine;
                if (nextStructureIndex != -1)
                {
                    endLine = sortedStructures[nextStructureIndex].Location.StartLine - 1;
                }
                else
                {
                    // If no next structure, use the end of the file
                    endLine = content.Split('\n').Length - 1;
                }

                // Update structure
                structure.Location.EndLine = endLine;
                structure.Size = endLine - structure.Location.StartLine + 1;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error calculating structure sizes");
        }
    }

    private static int GetLineNumber(string content, int position)
    {
        // Count newlines before the position
        return content[..position].Count(c => c == '\n');
    }

    private string GetNamespaceForPosition(List<CodeStructure> structures, int position, string content)
    {
        var lineNumber = GetLineNumber(content, position);

        // Find the namespace that contains this position
        var containingNamespace = structures
            .Where(s => s.Type == StructureType.Namespace)
            .Where(s => s.Location.StartLine <= lineNumber && s.Location.EndLine >= lineNumber)
            .OrderByDescending(s => s.Location.StartLine) // In case of nested namespaces, get the innermost one
            .FirstOrDefault();

        return containingNamespace?.Name ?? string.Empty;
    }

    private string GetClassForPosition(List<CodeStructure> structures, int position, string content)
    {
        var lineNumber = GetLineNumber(content, position);

        // Find the class that contains this position
        var containingClass = structures
            .Where(s => s.Type == StructureType.Class || s.Type == StructureType.Interface)
            .Where(s => s.Location.StartLine <= lineNumber && s.Location.EndLine >= lineNumber)
            .OrderByDescending(s => s.Location.StartLine) // In case of nested classes, get the innermost one
            .FirstOrDefault();

        return containingClass?.Name ?? string.Empty;
    }

    private string GetCodeSnippet(string content, int position)
    {
        try
        {
            var lines = content.Split('\n');
            var lineNumber = GetLineNumber(content, position);

            // Get a few lines before and after
            var startLine = Math.Max(0, lineNumber - 1);
            var endLine = Math.Min(lines.Length - 1, lineNumber + 1);

            var snippetLines = lines.Skip(startLine).Take(endLine - startLine + 1);
            return string.Join("\n", snippetLines);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting code snippet");
            return string.Empty;
        }
    }

    private static bool ParseOption(Dictionary<string, string>? options, string key, bool defaultValue)
    {
        if (options == null || !options.TryGetValue(key, out var value))
        {
            return defaultValue;
        }

        return value.Equals("true", StringComparison.OrdinalIgnoreCase);
    }
}
