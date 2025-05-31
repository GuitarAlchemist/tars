using System.Text.RegularExpressions;
using Microsoft.Extensions.Logging;
using TarsEngine.Models;
using TarsEngine.Models.Metrics;
using TarsEngine.Services.Interfaces;

namespace TarsEngine.Services;

/// <summary>
/// Analyzer for F# code
/// </summary>
public class FSharpAnalyzer : ILanguageAnalyzer
{
    private readonly ILogger<FSharpAnalyzer> _logger;
    private readonly CodeSmellDetector _codeSmellDetector;
    private readonly ComplexityAnalyzer _complexityAnalyzer;
    private readonly PerformanceAnalyzer _performanceAnalyzer;

    /// <summary>
    /// Initializes a new instance of the <see cref="FSharpAnalyzer"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    public FSharpAnalyzer(ILogger<FSharpAnalyzer> logger)
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
            var includeStructures = options?.GetValueOrDefault("IncludeStructures", "true").Equals("true", StringComparison.OrdinalIgnoreCase) ?? true;
            var includeMetrics = options?.GetValueOrDefault("IncludeMetrics", "true").Equals("true", StringComparison.OrdinalIgnoreCase) ?? true;
            var includeIssues = options?.GetValueOrDefault("IncludeIssues", "true").Equals("true", StringComparison.OrdinalIgnoreCase) ?? true;
            var analyzeComplexity = options?.GetValueOrDefault("AnalyzeComplexity", "true").Equals("true", StringComparison.OrdinalIgnoreCase) ?? true;
            var analyzeMaintainability = options?.GetValueOrDefault("AnalyzeMaintainability", "true").Equals("true", StringComparison.OrdinalIgnoreCase) ?? true;
            var analyzePerformance = options?.GetValueOrDefault("AnalyzePerformance", "true").Equals("true", StringComparison.OrdinalIgnoreCase) ?? true;
            var analyzeSecurity = options?.GetValueOrDefault("AnalyzeSecurity", "true").Equals("true", StringComparison.OrdinalIgnoreCase) ?? true;
            var analyzeStyle = options?.GetValueOrDefault("AnalyzeStyle", "true").Equals("true", StringComparison.OrdinalIgnoreCase) ?? true;

            // Extract structures
            if (includeStructures)
            {
                result.Structures.AddRange(ExtractStructures(content));
            }

            // Calculate metrics
            if (includeMetrics)
            {
                var metrics = await CalculateMetricsAsync(content, result.Structures, analyzeComplexity, analyzeMaintainability);
                result.Metrics.AddRange(metrics);
            }

            // Detect issues
            if (includeIssues)
            {
                // Detect code smells
                if (analyzeMaintainability || analyzeStyle)
                {
                    var codeSmells = _codeSmellDetector.DetectCodeSmells(content, "fsharp");
                    result.Issues.AddRange(codeSmells);
                }

                // Detect complexity issues
                if (analyzeComplexity)
                {
                    var complexityIssues = _complexityAnalyzer.DetectComplexityIssues(content, "fsharp", result.Structures);
                    result.Issues.AddRange(complexityIssues);
                }

                // Detect performance issues
                if (analyzePerformance)
                {
                    var performanceIssues = _performanceAnalyzer.DetectPerformanceIssues(content, "fsharp");
                    result.Issues.AddRange(performanceIssues);
                }

                // Detect security issues
                if (analyzeSecurity)
                {
                    var securityIssues = DetectSecurityIssues(content);
                    result.Issues.AddRange(securityIssues);
                }
            }

            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing F# code");
            return new CodeAnalysisResult
            {
                FilePath = "memory",
                ErrorMessage = ex.Message,
                Language = LanguageEnum,
                IsSuccessful = false,
                Errors = { ex.Message }
            };
        }
    }

    /// <inheritdoc/>
    public async Task<Dictionary<string, string>> GetAvailableOptionsAsync()
    {
        // Move the dictionary creation to a background thread since it could involve
        // loading configuration or computing dynamic options
        return await Task.Run(() => new Dictionary<string, string>
        {
            { "AnalyzePerformance", "Whether to analyze performance issues (true/false)" },
            { "AnalyzeSecurity", "Whether to analyze security issues (true/false)" },
            { "AnalyzeStyle", "Whether to analyze style issues (true/false)" },
            { "AnalyzeComplexity", "Whether to analyze code complexity (true/false)" },
            { "AnalyzeMaintainability", "Whether to analyze maintainability issues (true/false)" },
            { "IncludeMetrics", "Whether to include metrics in the analysis (true/false)" },
            { "IncludeStructures", "Whether to include code structures in the analysis (true/false)" },
            { "IncludeIssues", "Whether to include issues in the analysis (true/false)" },
            { "MaxIssues", "Maximum number of issues to report" },
            { "MinSeverity", "Minimum severity level for reported issues (Info/Warning/Error)" },
            { "AnalyzeDocumentation", "Whether to analyze XML documentation completeness (true/false)" },
            { "CheckUnusedBindings", "Whether to check for unused bindings (true/false)" },
            { "CheckMutableVariables", "Whether to check for mutable variables usage (true/false)" },
            { "CheckImperativeCode", "Whether to check for imperative code patterns (true/false)" }
        });
    }

    /// <inheritdoc/>
    public async Task<Dictionary<CodeIssueType, string>> GetLanguageSpecificIssueTypesAsync()
    {
        // Move the dictionary creation to a background thread since it could potentially
        // involve loading issue descriptions from resources or computing dynamic descriptions
        return await Task.Run(() => new Dictionary<CodeIssueType, string>
        {
            { CodeIssueType.CodeSmell, "F#-specific code smells like mutable variables, imperative loops, etc." },
            { CodeIssueType.Performance, "F#-specific performance issues like inefficient list operations, non-tail recursion, etc." },
            { CodeIssueType.Maintainability, "F#-specific maintainability issues like complex pattern matching, excessive type annotations, etc." },
            { CodeIssueType.Security, "F#-specific security issues like unsafe type casts, unverified external data, etc." },
            { CodeIssueType.Design, "F#-specific design issues like improper use of discriminated unions, record types, etc." }
        });
    }

    /// <inheritdoc/>
    public async Task<Dictionary<MetricType, string>> GetLanguageSpecificMetricTypesAsync()
    {
        // Move dictionary creation to a background thread since it could potentially
        // involve loading F#-specific metric descriptions or computing language-specific details
        return await Task.Run(() => new Dictionary<MetricType, string>
        {
            { MetricType.Complexity, "F#-specific complexity metrics including pattern matching complexity" },
            { MetricType.Size, "Code size metrics for F# modules and functions" },
            { MetricType.Coupling, "Module and type coupling metrics" },
            { MetricType.Cohesion, "Module and type cohesion metrics" },
            { MetricType.Inheritance, "Type hierarchy and inheritance metrics" },
            { MetricType.Maintainability, "F#-specific maintainability metrics" },
            { MetricType.Documentation, "XML documentation coverage metrics" },
            { MetricType.TestCoverage, "Unit test coverage metrics for F# code" }
        });
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

    private async Task<List<CodeMetric>> CalculateMetricsAsync(string content, List<CodeStructure> structures, bool analyzeComplexity, bool analyzeMaintainability)
    {
        return await Task.Run(() => {
            var metrics = new List<CodeMetric>();
            try
            {
                foreach (var structure in structures)
                {
                    if (analyzeComplexity)
                    {
                        metrics.Add(new CodeMetric
                        {
                            Name = "Complexity",
                            Value = CalculateComplexity(content, structure),
                            Type = MetricType.Complexity,  // Changed from string to enum
                            Location = structure.Location
                        });
                    }

                    if (analyzeMaintainability)
                    {
                        metrics.Add(new CodeMetric
                        {
                            Name = "Maintainability",
                            Value = CalculateMaintainability(content, structure),
                            Type = MetricType.Maintainability,  // Changed from Category string to Type enum
                            Location = structure.Location
                        });
                    }
                }

                return metrics;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error calculating metrics for F# code");
                return metrics;
            }
        });
    }

    private double CalculateComplexity(string content, CodeStructure structure)
    {
        // Implementation for calculating complexity
        return 0;
    }

    private double CalculateMaintainability(string content, CodeStructure structure)
    {
        // Implementation for calculating maintainability
        return 0;
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

    /// <inheritdoc/>
    public async Task<List<ReadabilityMetric>> AnalyzeReadabilityAsync(string filePath, string language, ReadabilityType readabilityType)
    {
        try
        {
            _logger.LogInformation("Analyzing readability for F# file {FilePath}", filePath);

            // Read file content asynchronously
            var fileContent = await File.ReadAllTextAsync(filePath);

            // Move CPU-bound readability analysis to a background thread
            return await Task.Run(() =>
            {
                var metrics = new List<ReadabilityMetric>();
                var fileName = Path.GetFileName(filePath);

                // Calculate basic readability metrics
                var linesOfCode = fileContent.Split('\n').Length;
                var words = fileContent.Split([' ', '\n', '\r', '\t'], StringSplitOptions.RemoveEmptyEntries);
                var characters = fileContent.Length;

                var metric = new ReadabilityMetric
                {
                    Name = $"Readability - {fileName}",
                    Type = readabilityType,
                    FilePath = filePath,
                    Language = language,
                    Target = fileName,
                    TargetType = TargetType.File,
                    Timestamp = DateTime.UtcNow
                };

                switch (readabilityType)
                {
                    case ReadabilityType.IdentifierQuality:
                        metric.Value = CalculateIdentifierQualityScore(fileContent);
                        break;

                    case ReadabilityType.CommentQuality:
                        metric.Value = CalculateCommentQualityScore(fileContent);
                        break;

                    case ReadabilityType.CodeStructure:
                        metric.Value = CalculateCodeStructureScore(fileContent);
                        break;

                    case ReadabilityType.Overall:
                        metric.Value = CalculateOverallReadabilityScore(fileContent);
                        break;

                    default:
                        metric.Value = 0;
                        _logger.LogWarning("Unsupported readability type: {ReadabilityType}", readabilityType);
                        break;
                }

                metrics.Add(metric);
                return metrics;
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing readability for F# file {FilePath}", filePath);
            return [];
        }
    }

    private double CalculateIdentifierQualityScore(string content)
    {
        // TODO: Implement identifier quality analysis
        return 0;
    }

    private double CalculateCommentQualityScore(string content)
    {
        // TODO: Implement comment quality analysis
        return 0;
    }

    private double CalculateCodeStructureScore(string content)
    {
        // TODO: Implement code structure analysis
        return 0;
    }

    private double CalculateOverallReadabilityScore(string content)
    {
        // TODO: Implement overall readability analysis
        return 0;
    }
}
