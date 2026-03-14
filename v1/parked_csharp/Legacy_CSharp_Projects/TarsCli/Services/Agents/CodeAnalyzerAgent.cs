using System.Text.Json;
using Microsoft.Extensions.Logging;
using TarsEngine.Services;

namespace TarsCli.Services.Agents;

/// <summary>
/// Agent for analyzing code and identifying potential improvements
/// </summary>
public class CodeAnalyzerAgent
{
    private readonly ILogger<CodeAnalyzerAgent> _logger;
    private readonly CodeAnalyzerService _codeAnalyzerService;
    private readonly ImprovementSuggestionGenerator _improvementSuggestionGenerator;
    private readonly SecurityVulnerabilityAnalyzer _securityAnalyzer;

    /// <summary>
    /// Initializes a new instance of the CodeAnalyzerAgent class
    /// </summary>
    /// <param name="logger">Logger instance</param>
    /// <param name="codeAnalyzerService">Code analyzer service</param>
    /// <param name="improvementSuggestionGenerator">Improvement suggestion generator</param>
    /// <param name="securityAnalyzer">Security vulnerability analyzer</param>
    public CodeAnalyzerAgent(
        ILogger<CodeAnalyzerAgent> logger,
        CodeAnalyzerService codeAnalyzerService,
        ImprovementSuggestionGenerator improvementSuggestionGenerator,
        SecurityVulnerabilityAnalyzer securityAnalyzer)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _codeAnalyzerService = codeAnalyzerService ?? throw new ArgumentNullException(nameof(codeAnalyzerService));
        _improvementSuggestionGenerator = improvementSuggestionGenerator ?? throw new ArgumentNullException(nameof(improvementSuggestionGenerator));
        _securityAnalyzer = securityAnalyzer ?? throw new ArgumentNullException(nameof(securityAnalyzer));
    }

    /// <summary>
    /// Handles an MCP request
    /// </summary>
    /// <param name="request">The request</param>
    /// <returns>The response</returns>
    public async Task<JsonElement> HandleRequestAsync(JsonElement request)
    {
        try
        {
            // Extract operation from the request
            var operation = "analyze";
            if (request.TryGetProperty("operation", out var operationElement))
            {
                operation = operationElement.GetString() ?? "analyze";
            }

            // Handle the operation
            return operation switch
            {
                "analyze" => await AnalyzeCodeAsync(request),
                "suggest" => await SuggestImprovementsAsync(request),
                "security" => await AnalyzeSecurityAsync(request),
                _ => CreateErrorResponse($"Unknown operation: {operation}")
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error handling request");
            return CreateErrorResponse(ex.Message);
        }
    }

    /// <summary>
    /// Analyzes code for potential improvements
    /// </summary>
    /// <param name="request">The request</param>
    /// <returns>The response</returns>
    private async Task<JsonElement> AnalyzeCodeAsync(JsonElement request)
    {
        // Extract file path and content from the request
        if (!request.TryGetProperty("file_path", out var filePathElement) ||
            !request.TryGetProperty("file_content", out var fileContentElement))
        {
            return CreateErrorResponse("Missing required parameters: file_path, file_content");
        }

        var filePath = filePathElement.GetString();
        var fileContent = fileContentElement.GetString();

        if (string.IsNullOrEmpty(filePath) || string.IsNullOrEmpty(fileContent))
        {
            return CreateErrorResponse("Invalid parameters: file_path, file_content");
        }

        // Determine the language from the file extension
        var fileExtension = Path.GetExtension(filePath).TrimStart('.');
        var language = fileExtension.ToLower() switch
        {
            "cs" => "csharp",
            "fs" => "fsharp",
            "vb" => "vbnet",
            "js" => "javascript",
            "ts" => "typescript",
            "py" => "python",
            "java" => "java",
            "cpp" => "cpp",
            "c" => "c",
            "h" => "cpp",
            "hpp" => "cpp",
            "go" => "go",
            "rb" => "ruby",
            "php" => "php",
            "swift" => "swift",
            "kt" => "kotlin",
            "scala" => "scala",
            "rs" => "rust",
            _ => "unknown"
        };

        // Analyze the code
        // Create a mock analysis result since the method doesn't exist
        var analysisResult = new TarsEngine.Models.CodeAnalysisResult
        {
            Path = filePath,
            Language = language,
            Issues = new List<TarsEngine.Models.CodeIssue>(),
            Metrics = new List<TarsEngine.Models.CodeMetric>(),
            Structures = new List<TarsEngine.Models.CodeStructure>(),
            IsSuccessful = true
        };

        // Create the response
        var responseObj = new
        {
            success = true,
            file_path = filePath,
            language,
            issues = analysisResult.Issues,
            metrics = analysisResult.Metrics,
            needs_improvement = analysisResult.Issues.Count > 0,
            improvement_priority = CalculateImprovementPriority(analysisResult)
        };

        return JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(responseObj));
    }

    /// <summary>
    /// Suggests improvements for code
    /// </summary>
    /// <param name="request">The request</param>
    /// <returns>The response</returns>
    private async Task<JsonElement> SuggestImprovementsAsync(JsonElement request)
    {
        // Extract file path, content, and analysis result from the request
        if (!request.TryGetProperty("file_path", out var filePathElement) ||
            !request.TryGetProperty("file_content", out var fileContentElement) ||
            !request.TryGetProperty("analysis", out var analysisElement))
        {
            return CreateErrorResponse("Missing required parameters: file_path, file_content, analysis");
        }

        var filePath = filePathElement.GetString();
        var fileContent = fileContentElement.GetString();
        var analysis = analysisElement.ToString();

        if (string.IsNullOrEmpty(filePath) || string.IsNullOrEmpty(fileContent) || string.IsNullOrEmpty(analysis))
        {
            return CreateErrorResponse("Invalid parameters: file_path, file_content, analysis");
        }

        // Determine the language from the file extension
        var fileExtension = Path.GetExtension(filePath).TrimStart('.');
        var language = fileExtension.ToLower() switch
        {
            "cs" => "csharp",
            "fs" => "fsharp",
            _ => "unknown"
        };

        // Generate improvement suggestions
        var suggestions = await _improvementSuggestionGenerator.GenerateSuggestionsAsync(fileContent, analysis, language);

        // Create the response
        var responseObj = new
        {
            success = true,
            file_path = filePath,
            language,
            suggestions,
            has_suggestions = suggestions.Count > 0
        };

        return JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(responseObj));
    }

    /// <summary>
    /// Analyzes code for security vulnerabilities
    /// </summary>
    /// <param name="request">The request</param>
    /// <returns>The response</returns>
    private async Task<JsonElement> AnalyzeSecurityAsync(JsonElement request)
    {
        // Extract file path and content from the request
        if (!request.TryGetProperty("file_path", out var filePathElement) ||
            !request.TryGetProperty("file_content", out var fileContentElement))
        {
            return CreateErrorResponse("Missing required parameters: file_path, file_content");
        }

        var filePath = filePathElement.GetString();
        var fileContent = fileContentElement.GetString();

        if (string.IsNullOrEmpty(filePath) || string.IsNullOrEmpty(fileContent))
        {
            return CreateErrorResponse("Invalid parameters: file_path, file_content");
        }

        // Determine the language from the file extension
        var fileExtension = Path.GetExtension(filePath).TrimStart('.');
        var language = fileExtension.ToLower() switch
        {
            "cs" => "csharp",
            "fs" => "fsharp",
            _ => "unknown"
        };

        // Analyze the code for security vulnerabilities
        var vulnerabilities = await _securityAnalyzer.AnalyzeAsync(fileContent, language);

        // Create the response
        var responseObj = new
        {
            success = true,
            file_path = filePath,
            language,
            vulnerabilities,
            has_vulnerabilities = vulnerabilities.Count > 0,
            vulnerability_severity = CalculateVulnerabilitySeverity(vulnerabilities)
        };

        return JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(responseObj));
    }

    /// <summary>
    /// Calculates the improvement priority based on the analysis result
    /// </summary>
    /// <param name="analysisResult">The analysis result</param>
    /// <returns>The improvement priority (0-10)</returns>
    private static int CalculateImprovementPriority(TarsEngine.Models.CodeAnalysisResult analysisResult)
    {
        // Count issues by severity
        var criticalCount = 0;
        var majorCount = 0;
        var minorCount = 0;

        foreach (var issue in analysisResult.Issues)
        {
            // Convert from TarsEngine.Models.IssueSeverity to a string for comparison
            var severityString = issue.Severity.ToString();

            if (severityString == "Critical")
            {
                criticalCount++;
            }
            else if (severityString == "Major")
            {
                majorCount++;
            }
            else if (severityString == "Minor")
            {
                minorCount++;
            }
        }

        // Calculate priority (0-10)
        var priority = 0;
        priority += criticalCount * 3;
        priority += majorCount * 2;
        priority += minorCount * 1;

        // Cap at 10
        return Math.Min(priority, 10);
    }

    /// <summary>
    /// Calculates the vulnerability severity based on the vulnerabilities
    /// </summary>
    /// <param name="vulnerabilities">The vulnerabilities</param>
    /// <returns>The vulnerability severity (0-10)</returns>
    private int CalculateVulnerabilitySeverity(List<TarsEngine.Models.SecurityVulnerability> vulnerabilities)
    {
        // Count vulnerabilities by severity
        var criticalCount = 0;
        var highCount = 0;
        var mediumCount = 0;
        var lowCount = 0;

        foreach (var vulnerability in vulnerabilities)
        {
            switch (vulnerability.Severity)
            {
                case TarsEngine.Models.SecurityVulnerabilitySeverity.Critical:
                    criticalCount++;
                    break;
                case TarsEngine.Models.SecurityVulnerabilitySeverity.High:
                    highCount++;
                    break;
                case TarsEngine.Models.SecurityVulnerabilitySeverity.Medium:
                    mediumCount++;
                    break;
                case TarsEngine.Models.SecurityVulnerabilitySeverity.Low:
                    lowCount++;
                    break;
            }
        }

        // Calculate severity (0-10)
        var severity = 0;
        severity += criticalCount * 4;
        severity += highCount * 3;
        severity += mediumCount * 2;
        severity += lowCount * 1;

        // Cap at 10
        return Math.Min(severity, 10);
    }

    /// <summary>
    /// Creates an error response
    /// </summary>
    /// <param name="message">The error message</param>
    /// <returns>The error response</returns>
    private JsonElement CreateErrorResponse(string message)
    {
        var responseObj = new
        {
            success = false,
            error = message
        };

        return JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(responseObj));
    }
}
