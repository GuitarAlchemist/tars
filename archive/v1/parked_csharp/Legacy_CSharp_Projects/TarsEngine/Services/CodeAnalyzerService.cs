using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
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
    private readonly ILoggerFactory _loggerFactory;
    private readonly Dictionary<string, ILanguageAnalyzer> _languageAnalyzers = new();
    private readonly Dictionary<MetricType, (double Good, double Acceptable, double Poor)> _metricThresholds = new();

    /// <summary>
    /// Initializes a new instance of the <see cref="CodeAnalyzerService"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    public CodeAnalyzerService(ILogger<CodeAnalyzerService> logger, ILoggerFactory loggerFactory)
    {
        _logger = logger;
        _loggerFactory = loggerFactory;
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
                var analysisResult = new CodeAnalysisResult
                {
                    FilePath = filePath,
                    ErrorMessage = "File does not exist",
                    IsSuccessful = false,
                    Errors = { "File does not exist" }
                };
                return analysisResult;
            }

            var language = GetLanguageFromFilePath(filePath);
            if (string.IsNullOrEmpty(language))
            {
                var analysisResult = new CodeAnalysisResult
                {
                    FilePath = filePath,
                    ErrorMessage = "Unsupported file type",
                    IsSuccessful = false,
                    Errors = { "Unsupported file type" }
                };
                return analysisResult;
            }

            var content = await File.ReadAllTextAsync(filePath);
            var result = await AnalyzeContentAsync(content, language, options);
            // Set FilePath property instead of Path

            // Ensure FilePath is also set for compatibility
            var filePathProperty = result.GetType().GetProperty("FilePath");
            if (filePathProperty != null)
            {
                filePathProperty.SetValue(result, filePath);
            }

            _logger.LogInformation("Completed analysis of file: {FilePath}. Found {IssueCount} issues, {MetricCount} metrics, {StructureCount} structures",
                filePath, result.Issues.Count, result.Metrics.Count, result.Structures.Count);

            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing file: {FilePath}", filePath);
            var result = new CodeAnalysisResult
            {
                FilePath = filePath,
                ErrorMessage = $"Error analyzing file: {ex.Message}",
                IsSuccessful = false,
                Errors = { $"Error analyzing file: {ex.Message}" }
            };

            // Ensure FilePath is also set for compatibility
            var filePathProperty = result.GetType().GetProperty("FilePath");
            if (filePathProperty != null)
            {
                filePathProperty.SetValue(result, filePath);
            }

            return result;
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
                _logger.LogError("Directory does not exist: {DirectoryPath}", directoryPath);
                return new List<CodeAnalysisResult>();
            }

            var filePatterns = filePattern.Split(';');
            var files = new List<string>();

            foreach (var pattern in filePatterns)
            {
                var searchOption = recursive ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;
                files.AddRange(Directory.GetFiles(directoryPath, pattern.Trim(), searchOption));
            }

            var analysisResults = new List<CodeAnalysisResult>();
            foreach (var file in files)
            {
                try
                {
                    var result = await AnalyzeFileAsync(file, options);
                    if (result != null)
                    {
                        analysisResults.Add(result);
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error analyzing file: {FilePath}", file);
                    analysisResults.Add(new CodeAnalysisResult
                    {
                        FilePath = file,
                        ErrorMessage = $"Error analyzing file: {ex.Message}",
                        IsSuccessful = false,
                        Errors = { ex.Message }
                    });
                }
            }

            return analysisResults;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing directory: {DirectoryPath}", directoryPath);
            return new List<CodeAnalysisResult>();
        }
    }

    /// <inheritdoc/>
    public async Task<CodeAnalysisResult> AnalyzeContentAsync(string content, string language, Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Analyzing content for language: {Language}", language);

            if (!_languageAnalyzers.TryGetValue(language.ToLowerInvariant(), out var analyzer))
            {
                return new CodeAnalysisResult
                {
                    FilePath = string.Empty,
                    ErrorMessage = $"Unsupported language: {language}",
                    Language = ProgrammingLanguage.Unknown,
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

            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing content for language: {Language}", language);
            return new CodeAnalysisResult
            {
                FilePath = string.Empty,
                ErrorMessage = ex.Message,
                Language = ProgrammingLanguage.Unknown,
                IsSuccessful = false,
                Errors = { ex.Message }
            };
        }
    }

    /// <inheritdoc/>
    public async Task<List<string>> GetSupportedLanguagesAsync()
    {
        try
        {
            _logger.LogInformation("Getting supported languages");
            
            // Since this is CPU-bound work, we'll move it to a background thread
            return await Task.Run(() => _languageAnalyzers.Keys.ToList());
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting supported languages");
            return new List<string>();
        }
    }

    /// <inheritdoc/>
    public async Task<Dictionary<string, string>> GetAvailableOptionsAsync()
    {
        try
        {
            _logger.LogInformation("Getting available analysis options");
            
            // Since this is CPU-bound work, we'll move it to a background thread
            return await Task.Run(() => new Dictionary<string, string>
            {
                { "IncludeMetrics", "Whether to include metrics in the analysis (true/false)" },
                { "IncludeStructures", "Whether to include structures in the analysis (true/false)" },
                { "IncludeIssues", "Whether to include issues in the analysis (true/false)" },
                { "MaxIssues", "Maximum number of issues to include in the result" },
                { "MinSeverity", "Minimum severity of issues to include (Blocker, Critical, Major, Minor, Info)" },
                { "IncludeCodeSnippets", "Whether to include code snippets in issues (true/false)" },
                { "IncludeSuggestedFixes", "Whether to include suggested fixes in issues (true/false)" },
                { "MaxMethodLength", "Maximum allowed method length in lines" },
                { "MaxClassLength", "Maximum allowed class length in lines" },
                { "MaxCyclomaticComplexity", "Maximum allowed cyclomatic complexity" },
                { "MaxCognitiveComplexity", "Maximum allowed cognitive complexity" },
                { "MaxParameters", "Maximum allowed number of parameters" },
                { "MaxNesting", "Maximum allowed nesting depth" },
                { "EnableSecurityAnalysis", "Enable security vulnerability analysis" },
                { "EnablePerformanceAnalysis", "Enable performance analysis" },
                { "EnableStyleAnalysis", "Enable code style analysis" }
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting available analysis options");
            return new Dictionary<string, string>();
        }
    }

    /// <inheritdoc/>
    public async Task<Dictionary<CodeIssueType, string>> GetAvailableIssueTypesAsync()
    {
        try
        {
            _logger.LogInformation("Getting available issue types");
            
            // Since this is CPU-bound work, we'll move it to a background thread
            return await Task.Run(() => new Dictionary<CodeIssueType, string>
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
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting available issue types");
            return new Dictionary<CodeIssueType, string>();
        }
    }

    /// <inheritdoc/>
    public async Task<Dictionary<MetricType, string>> GetAvailableMetricTypesAsync()
    {
        try
        {
            _logger.LogInformation("Getting available metric types");
            
            // Since this is CPU-bound work, we'll move it to a background thread
            return await Task.Run(() => new Dictionary<MetricType, string>
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
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting available metric types");
            return new Dictionary<MetricType, string>();
        }
    }

    /// <inheritdoc/>
    public async Task<Dictionary<MetricScope, string>> GetAvailableMetricScopesAsync()
    {
        try
        {
            _logger.LogInformation("Getting available metric scopes");
            
            // Since this is CPU-bound work, we'll move it to a background thread
            return await Task.Run(() => new Dictionary<MetricScope, string>
            {
                { MetricScope.Method, "Metrics that apply to individual methods" },
                { MetricScope.Class, "Metrics that apply to entire classes" },
                { MetricScope.File, "Metrics that apply to entire files" },
                { MetricScope.Namespace, "Metrics that apply to entire namespaces" },
                { MetricScope.Project, "Metrics that apply to the entire project" },
                { MetricScope.Solution, "Metrics that apply to the entire solution" }
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting available metric scopes");
            return new Dictionary<MetricScope, string>();
        }
    }

    /// <inheritdoc/>
    public async Task<Dictionary<MetricType, (double Good, double Acceptable, double Poor)>> GetMetricThresholdsAsync()
    {
        try
        {
            _logger.LogInformation("Getting metric thresholds");
            
            // Since this is CPU-bound work, we'll move it to a background thread
            return await Task.Run(() => new Dictionary<MetricType, (double Good, double Acceptable, double Poor)>(_metricThresholds));
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting metric thresholds");
            return new Dictionary<MetricType, (double Good, double Acceptable, double Poor)>();
        }
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

            // Since dictionary operations could potentially be slow for large dictionaries,
            // move the operation to a background thread
            return await Task.Run(() =>
            {
                _metricThresholds[metricType] = (goodThreshold, acceptableThreshold, poorThreshold);
                return true;
            });
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
                return [];
            }

            // Move the filtering operation to a background thread since it could be CPU-intensive for large result sets
            return await Task.Run(() => result.Issues
                .Where(i => i.Severity >= minSeverity)
                .Where(i => issueTypes == null || issueTypes.Contains(i.Type))
                .ToList());
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting issues for file: {FilePath}", filePath);
            return [];
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
                return [];
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
            return [];
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
                return [];
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
            return [];
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
        _languageAnalyzers["fsharp"] = new FSharpAnalyzer(_loggerFactory.CreateLogger<FSharpAnalyzer>());

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

    /// <inheritdoc/>
    public async Task<Dictionary<string, string>> GetAnalyzerOptionsAsync()
    {
        try
        {
            _logger.LogInformation("Getting analyzer options");
            
            // Move the CPU-bound dictionary creation to a background thread
            return await Task.Run(() => new Dictionary<string, string>
            {
                { "MaxMethodLength", "Maximum allowed method length in lines" },
                { "MaxClassLength", "Maximum allowed class length in lines" },
                { "MaxCyclomaticComplexity", "Maximum allowed cyclomatic complexity" },
                { "MaxCognitiveComplexity", "Maximum allowed cognitive complexity" },
                { "MaxParameters", "Maximum allowed number of parameters" },
                { "MaxNesting", "Maximum allowed nesting depth" },
                { "EnableSecurityAnalysis", "Enable security vulnerability analysis" },
                { "EnablePerformanceAnalysis", "Enable performance analysis" },
                { "EnableStyleAnalysis", "Enable code style analysis" },
                { "EnableDuplicationAnalysis", "Enable code duplication analysis" },
                { "EnableMaintainabilityAnalysis", "Enable maintainability analysis" },
                { "EnableDocumentationAnalysis", "Enable documentation analysis" },
                { "EnableTestCoverageAnalysis", "Enable test coverage analysis" },
                { "EnableDesignAnalysis", "Enable design pattern analysis" },
                { "EnableNamingAnalysis", "Enable naming convention analysis" }
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting analyzer options");
            return new Dictionary<string, string>();
        }
    }
}
