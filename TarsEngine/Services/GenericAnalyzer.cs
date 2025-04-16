using System.Text.RegularExpressions;
using Microsoft.Extensions.Logging;
using TarsEngine.Models;
using TarsEngine.Utilities;
using TarsEngine.Services.Interfaces;

namespace TarsEngine.Services;

/// <summary>
/// Generic analyzer for languages without specific implementations
/// </summary>
public class GenericAnalyzer : ILanguageAnalyzer
{
    private readonly ILogger _logger;
    private readonly string _language;
    private readonly CodeSmellDetector _codeSmellDetector;
    private readonly ComplexityAnalyzer _complexityAnalyzer;
    private readonly PerformanceAnalyzer _performanceAnalyzer;

    /// <summary>
    /// Initializes a new instance of the <see cref="GenericAnalyzer"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    /// <param name="language">The language to analyze</param>
    public GenericAnalyzer(ILogger logger, string language)
    {
        _logger = logger;
        _language = language;
        _codeSmellDetector = new CodeSmellDetector(logger);
        _complexityAnalyzer = new ComplexityAnalyzer(logger);
        _performanceAnalyzer = new PerformanceAnalyzer(logger);
    }

    /// <inheritdoc/>
    public string Language => _language;

    /// <summary>
    /// Gets the programming language enum value
    /// </summary>
    public ProgrammingLanguage LanguageEnum => ProgrammingLanguageConverter.FromString(_language);

    /// <inheritdoc/>
    public async Task<CodeAnalysisResult> AnalyzeAsync(string content, Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Analyzing {Language} code of length {Length}", Language, content?.Length ?? 0);

            // Move the CPU-intensive analysis work to a background thread
            return await Task.Run(() =>
            {
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

                // Extract structures (generic approach)
                if (includeStructures)
                {
                    result.Structures.AddRange(ExtractGenericStructures(content));
                }

                // Calculate metrics
                if (includeMetrics)
                {
                    result.Metrics.AddRange(CalculateGenericMetrics(content, result.Structures));
                }

                // Detect issues
                if (includeIssues)
                {
                    // Detect code smells
                    if (analyzeMaintainability || analyzeStyle)
                    {
                        result.Issues.AddRange(_codeSmellDetector.DetectCodeSmells(content, Language));
                    }

                    // Detect complexity issues
                    if (analyzeComplexity)
                    {
                        result.Issues.AddRange(_complexityAnalyzer.DetectComplexityIssues(content, Language, result.Structures));
                    }

                    // Detect performance issues
                    if (analyzePerformance)
                    {
                        result.Issues.AddRange(_performanceAnalyzer.DetectPerformanceIssues(content, Language));
                    }

                    // Detect security issues
                    if (analyzeSecurity)
                    {
                        result.Issues.AddRange(DetectGenericSecurityIssues(content));
                    }
                }

                _logger.LogInformation("Completed analysis of {Language} code. Found {IssueCount} issues, {MetricCount} metrics, {StructureCount} structures",
                    Language, result.Issues.Count, result.Metrics.Count, result.Structures.Count);

                return result;
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing {Language} code", Language);
            return new CodeAnalysisResult
            {
                FilePath = "memory",
                ErrorMessage = $"Error analyzing {Language} code: {ex.Message}",
                Language = LanguageEnum,
                IsSuccessful = false,
                Errors = { $"Error analyzing {Language} code: {ex.Message}" }
            };
        }
    }

    /// <inheritdoc/>
    public async Task<Dictionary<string, string>> GetAvailableOptionsAsync()
    {
        // Since building the options dictionary could potentially be CPU-intensive
        // when dealing with many options or complex option descriptions,
        // move it to a background thread
        return await Task.Run(() => new Dictionary<string, string>
        {
            { "IncludeMetrics", "Whether to include metrics in the analysis (true/false)" },
            { "IncludeStructures", "Whether to include structures in the analysis (true/false)" },
            { "IncludeIssues", "Whether to include issues in the analysis (true/false)" },
            { "MaxIssues", "Maximum number of issues to include in the result" },
            { "MinSeverity", "Minimum severity of issues to include (Blocker, Critical, Major, Minor, Info)" },
            { "AnalyzePerformance", "Whether to analyze performance issues (true/false)" },
            { "AnalyzeSecurity", "Whether to analyze security issues (true/false)" },
            { "AnalyzeStyle", "Whether to analyze style issues (true/false)" },
            { "AnalyzeComplexity", "Whether to analyze code complexity (true/false)" },
            { "AnalyzeMaintainability", "Whether to analyze maintainability issues (true/false)" }
        });
    }

    /// <inheritdoc/>
    public async Task<Dictionary<CodeIssueType, string>> GetLanguageSpecificIssueTypesAsync()
    {
        // Move dictionary creation to a background thread since it could potentially
        // involve loading issue descriptions or computing language-specific details
        return await Task.Run(() => new Dictionary<CodeIssueType, string>
        {
            { CodeIssueType.CodeSmell, "Generic code smells" },
            { CodeIssueType.Bug, "Potential bugs" },
            { CodeIssueType.Vulnerability, "Security vulnerabilities" },
            { CodeIssueType.SecurityHotspot, "Security hotspots" },
            { CodeIssueType.Performance, "Performance issues" },
            { CodeIssueType.Maintainability, "Maintainability issues" },
            { CodeIssueType.Design, "Design issues" },
            { CodeIssueType.Documentation, "Documentation issues" },
            { CodeIssueType.Duplication, "Code duplication" },
            { CodeIssueType.Complexity, "Complexity issues" },
            { CodeIssueType.Style, "Style issues" }
        });
    }

    /// <inheritdoc/>
    public async Task<Dictionary<MetricType, string>> GetLanguageSpecificMetricTypesAsync()
    {
        // Move dictionary creation to a background thread since it could potentially
        // involve loading metric descriptions or computing language-specific details
        return await Task.Run(() => new Dictionary<MetricType, string>
        {
            { MetricType.Complexity, "Complexity metrics" },
            { MetricType.Size, "Size metrics" },
            { MetricType.Coupling, "Coupling metrics" },
            { MetricType.Cohesion, "Cohesion metrics" },
            { MetricType.Inheritance, "Inheritance metrics" },
            { MetricType.Maintainability, "Maintainability metrics" },
            { MetricType.Documentation, "Documentation metrics" },
            { MetricType.TestCoverage, "Test coverage metrics" },
            { MetricType.Performance, "Performance metrics" }
        });
    }

    private List<CodeStructure> ExtractGenericStructures(string content)
    {
        var structures = new List<CodeStructure>();
        var lines = content.Split('\n');

        try
        {
            // Generic approach to find structures based on indentation and braces
            var braceStack = new Stack<(int Line, string Name, StructureType Type)>();
            var currentIndentation = 0;
            var inComment = false;

            for (var i = 0; i < lines.Length; i++)
            {
                var line = lines[i].TrimEnd();

                // Skip empty lines
                if (string.IsNullOrWhiteSpace(line))
                {
                    continue;
                }

                // Handle comments
                if (line.Trim().StartsWith("/*"))
                {
                    inComment = true;
                }

                if (inComment)
                {
                    if (line.Contains("*/"))
                    {
                        inComment = false;
                    }
                    continue;
                }

                if (line.Trim().StartsWith("//"))
                {
                    continue;
                }

                // Calculate indentation
                var indentation = line.Length - line.TrimStart().Length;

                // If indentation decreased, we might have exited a structure
                if (indentation < currentIndentation && braceStack.Count > 0)
                {
                    var (startLine, name, type) = braceStack.Pop();

                    // Add the structure
                    structures.Add(new CodeStructure
                    {
                        Type = type,
                        Name = name,
                        Location = new CodeLocation
                        {
                            StartLine = startLine,
                            EndLine = i - 1
                        },
                        Size = i - startLine
                    });
                }

                currentIndentation = indentation;

                // Try to identify structures based on common patterns
                if (line.Contains("{") && !line.Contains("}"))
                {
                    var structureName = "Unknown";
                    var structureType = StructureType.Other;

                    // Try to identify the structure type and name
                    if (line.Contains("class ") || line.Contains("interface ") || line.Contains("struct "))
                    {
                        structureType = line.Contains("class ") ? StructureType.Class :
                                       (line.Contains("interface ") ? StructureType.Interface : StructureType.Class);

                        var match = Regex.Match(line, @"(class|interface|struct)\s+([a-zA-Z0-9_]+)");
                        if (match.Success && match.Groups.Count > 2)
                        {
                            structureName = match.Groups[2].Value;
                        }
                    }
                    else if (line.Contains("function ") || line.Contains("def ") ||
                             Regex.IsMatch(line, @"(public|private|protected)\s+[a-zA-Z0-9_]+\s*\("))
                    {
                        structureType = StructureType.Method;

                        var match = Regex.Match(line, @"(function|def)\s+([a-zA-Z0-9_]+)|\s+([a-zA-Z0-9_]+)\s*\(");
                        if (match.Success)
                        {
                            structureName = match.Groups[2].Success ? match.Groups[2].Value :
                                           (match.Groups[3].Success ? match.Groups[3].Value : "Unknown");
                        }
                    }
                    else if (line.Contains("namespace ") || line.Contains("package "))
                    {
                        structureType = StructureType.Namespace;

                        var match = Regex.Match(line, @"(namespace|package)\s+([a-zA-Z0-9_.]+)");
                        if (match.Success && match.Groups.Count > 2)
                        {
                            structureName = match.Groups[2].Value;
                        }
                    }

                    braceStack.Push((i, structureName, structureType));
                }
            }

            // Handle any remaining structures
            while (braceStack.Count > 0)
            {
                var (startLine, name, type) = braceStack.Pop();

                structures.Add(new CodeStructure
                {
                    Type = type,
                    Name = name,
                    Location = new CodeLocation
                    {
                        StartLine = startLine,
                        EndLine = lines.Length - 1
                    },
                    Size = lines.Length - startLine
                });
            }

            return structures;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error extracting generic structures");
            return structures;
        }
    }

    private List<CodeMetric> CalculateGenericMetrics(string content, List<CodeStructure> structures)
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

            // Count non-empty, non-comment lines
            var nonEmptyLines = 0;
            var lines = content.Split('\n');
            var inComment = false;

            foreach (var line in lines)
            {
                var trimmedLine = line.Trim();

                if (string.IsNullOrWhiteSpace(trimmedLine))
                {
                    continue;
                }

                if (trimmedLine.StartsWith("/*"))
                {
                    inComment = true;
                }

                if (inComment)
                {
                    if (trimmedLine.Contains("*/"))
                    {
                        inComment = false;
                    }
                    continue;
                }

                if (trimmedLine.StartsWith("//"))
                {
                    continue;
                }

                nonEmptyLines++;
            }

            metrics.Add(new CodeMetric
            {
                Name = "Non-Empty Lines",
                Value = nonEmptyLines,
                Type = MetricType.Size,
                Scope = MetricScope.File
            });

            // Calculate metrics for each structure
            foreach (var structure in structures)
            {
                // Structure size
                metrics.Add(new CodeMetric
                {
                    Name = $"{structure.Type} Size",
                    Value = structure.Size,
                    Type = MetricType.Size,
                    Scope = structure.Type == StructureType.Method ? MetricScope.Method :
                           (structure.Type == StructureType.Class || structure.Type == StructureType.Interface ? MetricScope.Class : MetricScope.Namespace),
                    Target = structure.Name,
                    Location = structure.Location
                });
            }

            return metrics;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error calculating generic metrics");
            return metrics;
        }
    }

    private List<CodeIssue> DetectGenericSecurityIssues(string content)
    {
        var issues = new List<CodeIssue>();

        try
        {
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

            // Detect potential SQL injection
            var sqlInjectionRegex = new Regex(@"(sql|query)\s*=\s*[""'].*?\+\s*[^""']+\s*\+", RegexOptions.IgnoreCase | RegexOptions.Compiled);
            var sqlInjectionMatches = sqlInjectionRegex.Matches(content);
            foreach (Match match in sqlInjectionMatches)
            {
                issues.Add(new CodeIssue
                {
                    Type = CodeIssueType.Vulnerability,
                    Severity = TarsEngine.Models.IssueSeverity.Critical,
                    Title = "Potential SQL Injection",
                    Description = "String concatenation in SQL queries can lead to SQL injection vulnerabilities.",
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

            // Detect potential command injection
            var commandInjectionRegex = new Regex(@"(exec|system|spawn|eval|execute)\s*\([^)]*\+", RegexOptions.IgnoreCase | RegexOptions.Compiled);
            var commandInjectionMatches = commandInjectionRegex.Matches(content);
            foreach (Match match in commandInjectionMatches)
            {
                issues.Add(new CodeIssue
                {
                    Type = CodeIssueType.Vulnerability,
                    Severity = TarsEngine.Models.IssueSeverity.Critical,
                    Title = "Potential Command Injection",
                    Description = "String concatenation in command execution can lead to command injection vulnerabilities.",
                    Location = new CodeLocation
                    {
                        StartLine = GetLineNumber(content, match.Index)
                    },
                    CodeSnippet = GetCodeSnippet(content, match.Index),
                    SuggestedFix = "Use safe APIs for command execution and validate user input.",
                    ImpactScore = 0.9,
                    FixDifficultyScore = 0.4,
                    Tags = { "security", "command-injection" }
                });
            }

            return issues;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error detecting generic security issues");
            return issues;
        }
    }

    private int GetLineNumber(string content, int position)
    {
        // Count newlines before the position
        return content.Substring(0, position).Count(c => c == '\n');
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

    private bool ParseOption(Dictionary<string, string>? options, string key, bool defaultValue)
    {
        if (options == null || !options.TryGetValue(key, out var value))
        {
            return defaultValue;
        }

        return value.Equals("true", StringComparison.OrdinalIgnoreCase);
    }
}
