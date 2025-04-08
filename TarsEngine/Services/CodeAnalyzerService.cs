using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsEngine.Models;
using TarsEngine.Services.Interfaces;

namespace TarsEngine.Services;

/// <summary>
/// Service for analyzing code to identify improvement opportunities
/// </summary>
public class CodeAnalyzerService : ICodeAnalyzerService
{
    private readonly ILogger<CodeAnalyzerService> _logger;
    private readonly Dictionary<string, ILanguageAnalyzer> _languageAnalyzers = new();
    private readonly Dictionary<MetricType, (double Good, double Acceptable, double Poor)> _metricThresholds = new();

    /// <summary>
    /// Initializes a new instance of the <see cref="CodeAnalyzerService"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    public CodeAnalyzerService(ILogger<CodeAnalyzerService> logger)
    {
        _logger = logger;
        InitializeLanguageAnalyzers();
        InitializeMetricThresholds();
    }

    /// <inheritdoc/>
    public async Task<CodeAnalysisResult> AnalyzeFileAsync(string filePath, Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Analyzing file: {FilePath}", filePath);

            if (!File.Exists(filePath))
            {
                return new CodeAnalysisResult
                {
                    Path = filePath,
                    IsSuccessful = false,
                    Errors = { "File does not exist" }
                };
            }

            var language = GetLanguageFromFilePath(filePath);
            if (string.IsNullOrEmpty(language))
            {
                return new CodeAnalysisResult
                {
                    Path = filePath,
                    IsSuccessful = false,
                    Errors = { "Unsupported file type" }
                };
            }

            var content = await File.ReadAllTextAsync(filePath);
            var result = await AnalyzeContentAsync(content, language, options);
            result.Path = filePath;

            _logger.LogInformation("Completed analysis of file: {FilePath}. Found {IssueCount} issues, {MetricCount} metrics, {StructureCount} structures",
                filePath, result.Issues.Count, result.Metrics.Count, result.Structures.Count);

            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing file: {FilePath}", filePath);
            return new CodeAnalysisResult
            {
                Path = filePath,
                IsSuccessful = false,
                Errors = { $"Error analyzing file: {ex.Message}" }
            };
        }
    }

    /// <inheritdoc/>
    public async Task<List<CodeAnalysisResult>> AnalyzeDirectoryAsync(string directoryPath, bool recursive = true, string filePattern = "*.cs;*.fs", Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Analyzing directory: {DirectoryPath}, Recursive: {Recursive}, Pattern: {FilePattern}",
                directoryPath, recursive, filePattern);

            if (!Directory.Exists(directoryPath))
            {
                return new List<CodeAnalysisResult>
                {
                    new CodeAnalysisResult
                    {
                        Path = directoryPath,
                        IsSuccessful = false,
                        Errors = { "Directory does not exist" }
                    }
                };
            }

            var results = new List<CodeAnalysisResult>();
            var patterns = filePattern.Split(';');
            var searchOption = recursive ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;

            foreach (var pattern in patterns)
            {
                var files = Directory.GetFiles(directoryPath, pattern, searchOption);
                _logger.LogInformation("Found {FileCount} files matching pattern {Pattern}", files.Length, pattern);

                foreach (var file in files)
                {
                    var result = await AnalyzeFileAsync(file, options);
                    results.Add(result);
                }
            }

            _logger.LogInformation("Completed analysis of directory: {DirectoryPath}. Analyzed {FileCount} files",
                directoryPath, results.Count);

            return results;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing directory: {DirectoryPath}", directoryPath);
            return new List<CodeAnalysisResult>
            {
                new CodeAnalysisResult
                {
                    Path = directoryPath,
                    IsSuccessful = false,
                    Errors = { $"Error analyzing directory: {ex.Message}" }
                }
            };
        }
    }

    /// <inheritdoc/>
    public async Task<CodeAnalysisResult> AnalyzeContentAsync(string content, string language, Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Analyzing content of language: {Language}, Length: {ContentLength}", language, content?.Length ?? 0);

            if (string.IsNullOrWhiteSpace(content))
            {
                return new CodeAnalysisResult
                {
                    Language = language,
                    IsSuccessful = false,
                    Errors = { "Content is empty or whitespace" }
                };
            }

            if (!_languageAnalyzers.TryGetValue(language.ToLowerInvariant(), out var analyzer))
            {
                return new CodeAnalysisResult
                {
                    Language = language,
                    IsSuccessful = false,
                    Errors = { $"Unsupported language: {language}" }
                };
            }

            var result = await analyzer.AnalyzeAsync(content, options);

            // Apply metric thresholds
            foreach (var metric in result.Metrics)
            {
                if (_metricThresholds.TryGetValue(metric.Type, out var thresholds))
                {
                    metric.GoodThreshold = thresholds.Good;
                    metric.AcceptableThreshold = thresholds.Acceptable;
                    metric.PoorThreshold = thresholds.Poor;
                }
            }

            _logger.LogInformation("Completed analysis of content. Found {IssueCount} issues, {MetricCount} metrics, {StructureCount} structures",
                result.Issues.Count, result.Metrics.Count, result.Structures.Count);

            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing content of language: {Language}", language);
            return new CodeAnalysisResult
            {
                Language = language,
                IsSuccessful = false,
                Errors = { $"Error analyzing content: {ex.Message}" }
            };
        }
    }

    /// <inheritdoc/>
    public async Task<List<string>> GetSupportedLanguagesAsync()
    {
        return _languageAnalyzers.Keys.ToList();
    }

    /// <inheritdoc/>
    public async Task<Dictionary<string, string>> GetAvailableOptionsAsync()
    {
        return new Dictionary<string, string>
        {
            { "IncludeMetrics", "Whether to include metrics in the analysis (true/false)" },
            { "IncludeStructures", "Whether to include structures in the analysis (true/false)" },
            { "IncludeIssues", "Whether to include issues in the analysis (true/false)" },
            { "MaxIssues", "Maximum number of issues to include in the result" },
            { "MinSeverity", "Minimum severity of issues to include (Blocker, Critical, Major, Minor, Info)" },
            { "IncludeCodeSnippets", "Whether to include code snippets in issues (true/false)" },
            { "IncludeSuggestedFixes", "Whether to include suggested fixes in issues (true/false)" },
            { "AnalyzePerformance", "Whether to analyze performance issues (true/false)" },
            { "AnalyzeComplexity", "Whether to analyze complexity issues (true/false)" },
            { "AnalyzeMaintainability", "Whether to analyze maintainability issues (true/false)" },
            { "AnalyzeSecurity", "Whether to analyze security issues (true/false)" },
            { "AnalyzeStyle", "Whether to analyze style issues (true/false)" }
        };
    }

    /// <inheritdoc/>
    public async Task<Dictionary<CodeIssueType, string>> GetAvailableIssueTypesAsync()
    {
        return new Dictionary<CodeIssueType, string>
        {
            { CodeIssueType.CodeSmell, "Code that works but may have maintainability issues" },
            { CodeIssueType.Bug, "Code that is likely to cause runtime errors or incorrect behavior" },
            { CodeIssueType.Vulnerability, "Code that may introduce security vulnerabilities" },
            { CodeIssueType.SecurityHotspot, "Code that should be reviewed for security concerns" },
            { CodeIssueType.Performance, "Code that may cause performance issues" },
            { CodeIssueType.Maintainability, "Code that may be difficult to maintain" },
            { CodeIssueType.Design, "Code that violates design principles" },
            { CodeIssueType.Documentation, "Code with missing or inadequate documentation" },
            { CodeIssueType.Duplication, "Duplicated code that should be refactored" },
            { CodeIssueType.Complexity, "Code with excessive complexity" },
            { CodeIssueType.Style, "Code that violates style guidelines" },
            { CodeIssueType.Naming, "Code with poor naming conventions" },
            { CodeIssueType.UnusedCode, "Code that is never used" },
            { CodeIssueType.DeadCode, "Code that can never be executed" },
            { CodeIssueType.Other, "Other issues not covered by the above categories" }
        };
    }

    /// <inheritdoc/>
    public async Task<Dictionary<MetricType, string>> GetAvailableMetricTypesAsync()
    {
        return new Dictionary<MetricType, string>
        {
            { MetricType.Complexity, "Measures the complexity of code (e.g., cyclomatic complexity)" },
            { MetricType.Size, "Measures the size of code (e.g., lines of code)" },
            { MetricType.Coupling, "Measures the coupling between components" },
            { MetricType.Cohesion, "Measures the cohesion within components" },
            { MetricType.Inheritance, "Measures the inheritance depth and breadth" },
            { MetricType.Maintainability, "Measures the maintainability of code" },
            { MetricType.Documentation, "Measures the documentation coverage" },
            { MetricType.TestCoverage, "Measures the test coverage" },
            { MetricType.Performance, "Measures the performance characteristics" },
            { MetricType.Other, "Other metrics not covered by the above categories" }
        };
    }

    /// <inheritdoc/>
    public async Task<Dictionary<MetricType, (double Good, double Acceptable, double Poor)>> GetMetricThresholdsAsync()
    {
        return new Dictionary<MetricType, (double Good, double Acceptable, double Poor)>(_metricThresholds);
    }

    /// <inheritdoc/>
    public async Task<bool> SetMetricThresholdsAsync(MetricType metricType, double goodThreshold, double acceptableThreshold, double poorThreshold)
    {
        try
        {
            _logger.LogInformation("Setting thresholds for metric type {MetricType}: Good={Good}, Acceptable={Acceptable}, Poor={Poor}",
                metricType, goodThreshold, acceptableThreshold, poorThreshold);

            // Validate thresholds
            if (goodThreshold > acceptableThreshold || acceptableThreshold > poorThreshold)
            {
                _logger.LogWarning("Invalid thresholds for metric type {MetricType}: Good={Good}, Acceptable={Acceptable}, Poor={Poor}",
                    metricType, goodThreshold, acceptableThreshold, poorThreshold);
                return false;
            }

            _metricThresholds[metricType] = (goodThreshold, acceptableThreshold, poorThreshold);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error setting thresholds for metric type {MetricType}", metricType);
            return false;
        }
    }

    /// <inheritdoc/>
    public async Task<List<CodeIssue>> GetIssuesForFileAsync(string filePath, List<CodeIssueType>? issueTypes = null, TarsEngine.Models.IssueSeverity minSeverity = TarsEngine.Models.IssueSeverity.Info, Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Getting issues for file: {FilePath}", filePath);

            var result = await AnalyzeFileAsync(filePath, options);
            if (!result.IsSuccessful)
            {
                _logger.LogWarning("Analysis of file {FilePath} was not successful", filePath);
                return new List<CodeIssue>();
            }

            var issues = result.Issues
                .Where(i => i.Severity >= minSeverity)
                .Where(i => issueTypes == null || issueTypes.Contains(i.Type))
                .ToList();

            _logger.LogInformation("Found {IssueCount} issues for file {FilePath}", issues.Count, filePath);
            return issues;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting issues for file: {FilePath}", filePath);
            return new List<CodeIssue>();
        }
    }

    /// <inheritdoc/>
    public async Task<List<CodeMetric>> GetMetricsForFileAsync(string filePath, List<MetricType>? metricTypes = null, MetricScope? scope = null, Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Getting metrics for file: {FilePath}", filePath);

            var result = await AnalyzeFileAsync(filePath, options);
            if (!result.IsSuccessful)
            {
                _logger.LogWarning("Analysis of file {FilePath} was not successful", filePath);
                return new List<CodeMetric>();
            }

            var metrics = result.Metrics
                .Where(m => metricTypes == null || metricTypes.Contains(m.Type))
                .Where(m => scope == null || m.Scope == scope)
                .ToList();

            _logger.LogInformation("Found {MetricCount} metrics for file {FilePath}", metrics.Count, filePath);
            return metrics;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting metrics for file: {FilePath}", filePath);
            return new List<CodeMetric>();
        }
    }

    /// <inheritdoc/>
    public async Task<List<CodeStructure>> GetStructuresForFileAsync(string filePath, List<StructureType>? structureTypes = null, Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Getting structures for file: {FilePath}", filePath);

            var result = await AnalyzeFileAsync(filePath, options);
            if (!result.IsSuccessful)
            {
                _logger.LogWarning("Analysis of file {FilePath} was not successful", filePath);
                return new List<CodeStructure>();
            }

            var structures = result.Structures
                .Where(s => structureTypes == null || structureTypes.Contains(s.Type))
                .ToList();

            _logger.LogInformation("Found {StructureCount} structures for file {FilePath}", structures.Count, filePath);
            return structures;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting structures for file: {FilePath}", filePath);
            return new List<CodeStructure>();
        }
    }

    private string GetLanguageFromFilePath(string filePath)
    {
        var extension = Path.GetExtension(filePath).ToLowerInvariant();
        return extension switch
        {
            ".cs" => "csharp",
            ".fs" => "fsharp",
            ".js" => "javascript",
            ".ts" => "typescript",
            ".py" => "python",
            ".java" => "java",
            ".rb" => "ruby",
            ".php" => "php",
            ".go" => "go",
            ".rs" => "rust",
            ".c" => "c",
            ".cpp" => "cpp",
            ".h" => "c",
            ".hpp" => "cpp",
            ".swift" => "swift",
            ".kt" => "kotlin",
            ".scala" => "scala",
            ".m" => "objective-c",
            ".sql" => "sql",
            ".html" => "html",
            ".css" => "css",
            ".xml" => "xml",
            ".json" => "json",
            ".yaml" => "yaml",
            ".yml" => "yaml",
            ".md" => "markdown",
            _ => string.Empty
        };
    }

    private void InitializeLanguageAnalyzers()
    {
        // Register language analyzers
        _languageAnalyzers["csharp"] = new CSharpAnalyzer(_logger);
        _languageAnalyzers["fsharp"] = new FSharpAnalyzer(_logger);

        // Add placeholder analyzers for other languages
        _languageAnalyzers["javascript"] = new GenericAnalyzer(_logger, "javascript");
        _languageAnalyzers["typescript"] = new GenericAnalyzer(_logger, "typescript");
        _languageAnalyzers["python"] = new GenericAnalyzer(_logger, "python");
        _languageAnalyzers["java"] = new GenericAnalyzer(_logger, "java");
    }

    private void InitializeMetricThresholds()
    {
        // Initialize default metric thresholds
        _metricThresholds[MetricType.Complexity] = (10, 20, 30);
        _metricThresholds[MetricType.Size] = (100, 500, 1000);
        _metricThresholds[MetricType.Coupling] = (5, 10, 15);
        _metricThresholds[MetricType.Cohesion] = (0.8, 0.6, 0.4);
        _metricThresholds[MetricType.Inheritance] = (3, 5, 7);
        _metricThresholds[MetricType.Maintainability] = (80, 60, 40);
        _metricThresholds[MetricType.Documentation] = (0.8, 0.5, 0.2);
        _metricThresholds[MetricType.TestCoverage] = (0.8, 0.6, 0.4);
        _metricThresholds[MetricType.Performance] = (0.8, 0.6, 0.4);
    }
}
