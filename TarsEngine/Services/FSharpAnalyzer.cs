using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsEngine.Models;
using TarsEngine.Services.Interfaces;
using TarsEngine.Utilities;

namespace TarsEngine.Services;

/// <summary>
/// Analyzer for F# code
/// </summary>
public class FSharpAnalyzer : ILanguageAnalyzer
{
    private readonly ILogger _logger;
    private readonly CodeSmellDetector _codeSmellDetector;
    private readonly ComplexityAnalyzer _complexityAnalyzer;
    private readonly PerformanceAnalyzer _performanceAnalyzer;

    /// <summary>
    /// Initializes a new instance of the <see cref="FSharpAnalyzer"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    public FSharpAnalyzer(ILogger logger)
    {
        _logger = logger;
        _codeSmellDetector = new CodeSmellDetector(logger);
        _complexityAnalyzer = new ComplexityAnalyzer(logger);
        _performanceAnalyzer = new PerformanceAnalyzer(logger);
    }

    /// <inheritdoc/>
    public string Language => "fsharp";

    /// <summary>
    /// Gets the programming language enum value
    /// </summary>
    public static ProgrammingLanguage LanguageEnum => ProgrammingLanguage.FSharp;

    /// <inheritdoc/>
    public async Task<CodeAnalysisResult> AnalyzeAsync(string content, Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Analyzing F# code of length {Length}", content?.Length ?? 0);

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

            // Extract structures (modules, types, functions, etc.)
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
                    result.Issues.AddRange(_codeSmellDetector.DetectCodeSmells(content, "fsharp"));
                }

                // Detect complexity issues
                if (analyzeComplexity)
                {
                    result.Issues.AddRange(_complexityAnalyzer.DetectComplexityIssues(content, "fsharp", result.Structures));
                }

                // Detect performance issues
                if (analyzePerformance)
                {
                    result.Issues.AddRange(_performanceAnalyzer.DetectPerformanceIssues(content, "fsharp"));
                }

                // Detect security issues
                if (analyzeSecurity)
                {
                    result.Issues.AddRange(DetectSecurityIssues(content));
                }
            }

            _logger.LogInformation("Completed analysis of F# code. Found {IssueCount} issues, {MetricCount} metrics, {StructureCount} structures",
                result.Issues.Count, result.Metrics.Count, result.Structures.Count);

            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing F# code");
            return new CodeAnalysisResult
            {
                FilePath = "memory",
                ErrorMessage = $"Error analyzing F# code: {ex.Message}",
                Language = LanguageEnum,
                IsSuccessful = false,
                Errors = { $"Error analyzing F# code: {ex.Message}" }
            };
        }
    }

    /// <inheritdoc/>
    public async Task<Dictionary<string, string>> GetAvailableOptionsAsync()
    {
        return new Dictionary<string, string>
        {
            { "AnalyzeTypeInference", "Whether to analyze type inference (true/false)" },
            { "AnalyzePatternMatching", "Whether to analyze pattern matching (true/false)" },
            { "AnalyzeRecursion", "Whether to analyze recursion (true/false)" },
            { "AnalyzeHigherOrderFunctions", "Whether to analyze higher-order functions (true/false)" },
            { "AnalyzeComputationExpressions", "Whether to analyze computation expressions (true/false)" },
            { "AnalyzeActivePatterns", "Whether to analyze active patterns (true/false)" },
            { "AnalyzeUnits", "Whether to analyze units of measure (true/false)" },
            { "AnalyzeTypeProviders", "Whether to analyze type providers (true/false)" },
            { "AnalyzeAsyncCode", "Whether to analyze async code (true/false)" },
            { "AnalyzeInterop", "Whether to analyze .NET interop (true/false)" }
        };
    }

    /// <inheritdoc/>
    public async Task<Dictionary<CodeIssueType, string>> GetLanguageSpecificIssueTypesAsync()
    {
        return new Dictionary<CodeIssueType, string>
        {
            { CodeIssueType.CodeSmell, "F#-specific code smells like mutable variables, imperative loops, etc." },
            { CodeIssueType.Performance, "F#-specific performance issues like inefficient list operations, non-tail recursion, etc." },
            { CodeIssueType.Maintainability, "F#-specific maintainability issues like complex pattern matching, excessive type annotations, etc." },
            { CodeIssueType.Security, "F#-specific security issues like unsafe type casts, unverified external data, etc." },
            { CodeIssueType.Design, "F#-specific design issues like improper use of discriminated unions, record types, etc." }
        };
    }

    /// <inheritdoc/>
    public async Task<Dictionary<MetricType, string>> GetLanguageSpecificMetricTypesAsync()
    {
        return new Dictionary<MetricType, string>
        {
            { MetricType.Complexity, "F#-specific complexity metrics like pattern matching complexity, recursion depth, etc." },
            { MetricType.Coupling, "F#-specific coupling metrics like module dependencies, function composition, etc." },
            { MetricType.Cohesion, "F#-specific cohesion metrics like module cohesion, type cohesion, etc." },
            { MetricType.Inheritance, "F#-specific inheritance metrics like interface implementation, abstract class usage, etc." }
        };
    }

    private List<CodeStructure> ExtractStructures(string content)
    {
        var structures = new List<CodeStructure>();

        try
        {
            // Extract modules
            var moduleRegex = new Regex(@"module\s+([a-zA-Z0-9_.]+)(?:\s*=)?", RegexOptions.Compiled);
            var moduleMatches = moduleRegex.Matches(content);
            foreach (Match match in moduleMatches)
            {
                if (match.Groups.Count > 1)
                {
                    var moduleName = match.Groups[1].Value;
                    structures.Add(new CodeStructure
                    {
                        Type = StructureType.Namespace, // Using Namespace type for modules
                        Name = moduleName,
                        Location = new CodeLocation
                        {
                            StartLine = GetLineNumber(content, match.Index),
                            Namespace = moduleName
                        }
                    });
                }
            }

            // Extract types (records, discriminated unions, classes)
            var typeRegex = new Regex(@"type\s+([a-zA-Z0-9_]+)(?:<[^>]+>)?(?:\s*=|\s*\(|\s*:)", RegexOptions.Compiled);
            var typeMatches = typeRegex.Matches(content);
            foreach (Match match in typeMatches)
            {
                if (match.Groups.Count > 1)
                {
                    var typeName = match.Groups[1].Value;
                    var namespaceName = GetNamespaceForPosition(structures, match.Index, content);
                    structures.Add(new CodeStructure
                    {
                        Type = StructureType.Class, // Using Class type for all F# types
                        Name = typeName,
                        Location = new CodeLocation
                        {
                            StartLine = GetLineNumber(content, match.Index),
                            Namespace = namespaceName,
                            ClassName = typeName
                        }
                    });
                }
            }

            // Extract functions
            var functionRegex = new Regex(@"let\s+(rec\s+)?([a-zA-Z0-9_]+)(?:\s+[a-zA-Z0-9_]+)*\s*=", RegexOptions.Compiled);
            var functionMatches = functionRegex.Matches(content);
            foreach (Match match in functionMatches)
            {
                if (match.Groups.Count > 2)
                {
                    var functionName = match.Groups[2].Value;
                    var isRecursive = match.Groups[1].Success;
                    var namespaceName = GetNamespaceForPosition(structures, match.Index, content);
                    var className = GetClassForPosition(structures, match.Index, content);

                    structures.Add(new CodeStructure
                    {
                        Type = StructureType.Method, // Using Method type for functions
                        Name = functionName,
                        ParentName = className,
                        Location = new CodeLocation
                        {
                            StartLine = GetLineNumber(content, match.Index),
                            Namespace = namespaceName,
                            ClassName = className,
                            MethodName = functionName
                        },
                        Properties = { { "IsRecursive", isRecursive.ToString() } }
                    });
                }
            }

            // Calculate structure sizes and update end lines
            CalculateStructureSizes(structures, content);

            return structures;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error extracting structures from F# code");
            return structures;
        }
    }

    private List<CodeMetric> CalculateMetrics(string content, List<CodeStructure> structures, bool analyzeComplexity, bool _)
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
                    case StructureType.Namespace: // Module
                        // Module size
                        metrics.Add(new CodeMetric
                        {
                            Name = "Module Size",
                            Value = structure.Size,
                            Type = MetricType.Size,
                            Scope = MetricScope.Namespace,
                            Target = structure.Name,
                            Location = structure.Location
                        });
                        break;

                    case StructureType.Class: // Type
                        // Type size
                        metrics.Add(new CodeMetric
                        {
                            Name = "Type Size",
                            Value = structure.Size,
                            Type = MetricType.Size,
                            Scope = MetricScope.Class,
                            Target = structure.Name,
                            Location = structure.Location
                        });

                        // Type complexity
                        if (analyzeComplexity)
                        {
                            var typeComplexity = CalculateTypeComplexity(content, structure);
                            metrics.Add(new CodeMetric
                            {
                                Name = "Type Complexity",
                                Value = typeComplexity,
                                Type = MetricType.Complexity,
                                Scope = MetricScope.Class,
                                Target = structure.Name,
                                Location = structure.Location
                            });
                        }
                        break;

                    case StructureType.Method: // Function
                        // Function size
                        metrics.Add(new CodeMetric
                        {
                            Name = "Function Size",
                            Value = structure.Size,
                            Type = MetricType.Size,
                            Scope = MetricScope.Method,
                            Target = structure.Name,
                            Location = structure.Location
                        });

                        // Function complexity
                        if (analyzeComplexity)
                        {
                            var cyclomaticComplexity = _complexityAnalyzer.CalculateMethodComplexity(content, structure);
                            metrics.Add(new CodeMetric
                            {
                                Name = "Cyclomatic Complexity",
                                Value = cyclomaticComplexity,
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

                            // Pattern matching complexity
                            var patternMatchingComplexity = CalculatePatternMatchingComplexity(content, structure);
                            metrics.Add(new CodeMetric
                            {
                                Name = "Pattern Matching Complexity",
                                Value = patternMatchingComplexity,
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
            _logger.LogError(ex, "Error calculating metrics for F# code");
            return metrics;
        }
    }

    private double CalculateTypeComplexity(string content, CodeStructure structure)
    {
        try
        {
            // Extract the type content
            var typeContent = ExtractStructureContent(content, structure);

            double complexity = 0;

            // Count union cases
            var unionCaseRegex = new Regex(@"\|\s*[a-zA-Z0-9_]+", RegexOptions.Compiled);
            complexity += unionCaseRegex.Matches(typeContent).Count * 0.5;

            // Count record fields
            var recordFieldRegex = new Regex(@"[a-zA-Z0-9_]+\s*:\s*[a-zA-Z0-9_<>]+", RegexOptions.Compiled);
            complexity += recordFieldRegex.Matches(typeContent).Count * 0.3;

            // Count generic type parameters
            var genericParamRegex = new Regex(@"<[^>]+>", RegexOptions.Compiled);
            complexity += genericParamRegex.Matches(typeContent).Count * 1.0;

            // Count interface implementations
            var interfaceRegex = new Regex(@"interface\s+[a-zA-Z0-9_<>]+", RegexOptions.Compiled);
            complexity += interfaceRegex.Matches(typeContent).Count * 1.5;

            // Count member definitions
            var memberRegex = new Regex(@"member\s+[a-zA-Z0-9_]+\.", RegexOptions.Compiled);
            complexity += memberRegex.Matches(typeContent).Count * 0.7;

            return complexity;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error calculating type complexity for {TypeName}", structure.Name);
            return 0;
        }
    }

    private double CalculatePatternMatchingComplexity(string content, CodeStructure structure)
    {
        try
        {
            // Extract the function content
            var functionContent = ExtractStructureContent(content, structure);

            double complexity = 0;

            // Count match expressions
            var matchRegex = new Regex(@"\bmatch\b", RegexOptions.Compiled);
            complexity += matchRegex.Matches(functionContent).Count * 1.0;

            // Count pattern cases
            var caseRegex = new Regex(@"\|\s*[^-]+\s*->", RegexOptions.Compiled);
            complexity += caseRegex.Matches(functionContent).Count * 0.5;

            // Count active patterns
            var activePatternRegex = new Regex(@"\(\|[^|]+\|\)", RegexOptions.Compiled);
            complexity += activePatternRegex.Matches(functionContent).Count * 1.5;

            // Count nested patterns (simplified)
            var nestedPatternRegex = new Regex(@"match\s+[^w]+\s+with\s+[^m]+match", RegexOptions.Compiled);
            complexity += nestedPatternRegex.Matches(functionContent).Count * 2.0;

            return complexity;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error calculating pattern matching complexity for {FunctionName}", structure.Name);
            return 0;
        }
    }

    private List<CodeIssue> DetectSecurityIssues(string content)
    {
        var issues = new List<CodeIssue>();

        try
        {
            // Detect unsafe code
            var unsafeRegex = new Regex(@"NativeInterop\.NativePtr|Microsoft\.FSharp\.NativeInterop|fixed\s+[a-zA-Z0-9_]+", RegexOptions.Compiled);
            var unsafeMatches = unsafeRegex.Matches(content);
            foreach (Match match in unsafeMatches)
            {
                issues.Add(new CodeIssue
                {
                    Type = CodeIssueType.SecurityHotspot,
                    Severity = TarsEngine.Models.IssueSeverity.Critical,
                    Title = "Unsafe Code Usage",
                    Description = "Using unsafe code can lead to memory corruption and security vulnerabilities.",
                    Location = new CodeLocation
                    {
                        StartLine = GetLineNumber(content, match.Index)
                    },
                    CodeSnippet = GetCodeSnippet(content, match.Index),
                    SuggestedFix = "Consider using safe alternatives if possible.",
                    ImpactScore = 0.8,
                    FixDifficultyScore = 0.7,
                    Tags = { "security", "unsafe-code" }
                });
            }

            // Detect unverified external data
            var externalDataRegex = new Regex(@"System\.IO\.File\.ReadAllText|System\.Net\.WebClient|HttpClient\.GetStringAsync", RegexOptions.Compiled);
            var externalDataMatches = externalDataRegex.Matches(content);
            foreach (Match match in externalDataMatches)
            {
                issues.Add(new CodeIssue
                {
                    Type = CodeIssueType.SecurityHotspot,
                    Severity = TarsEngine.Models.IssueSeverity.Major,
                    Title = "Unverified External Data",
                    Description = "Reading external data without validation can lead to security vulnerabilities.",
                    Location = new CodeLocation
                    {
                        StartLine = GetLineNumber(content, match.Index)
                    },
                    CodeSnippet = GetCodeSnippet(content, match.Index),
                    SuggestedFix = "Validate and sanitize external data before using it.",
                    ImpactScore = 0.6,
                    FixDifficultyScore = 0.4,
                    Tags = { "security", "input-validation" }
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
            _logger.LogError(ex, "Error detecting security issues in F# code");
            return issues;
        }
    }

    private string ExtractStructureContent(string content, CodeStructure structure)
    {
        try
        {
            // Get the lines for the structure
            var lines = content.Split('\n');
            var startLine = structure.Location.StartLine;
            var endLine = structure.Location.EndLine;

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
            for (int i = 0; i < sortedStructures.Count; i++)
            {
                var structure = sortedStructures[i];

                // Find the next structure at the same or higher level
                var nextStructureIndex = -1;
                for (int j = i + 1; j < sortedStructures.Count; j++)
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
                        (structure.Type == StructureType.Class && nextStructure.Type == StructureType.Class))
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
            .Where(s => s.Type == StructureType.Class)
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
