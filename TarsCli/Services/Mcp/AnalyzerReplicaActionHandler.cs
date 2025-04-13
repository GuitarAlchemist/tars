using System.Text.Json;
using TarsCli.Models;
using TarsCli.Services.CodeAnalysis;
using TarsCli.Services.Adapters;

namespace TarsCli.Services.Mcp;

/// <summary>
/// Action handler for analyzer replica
/// </summary>
public class AnalyzerReplicaActionHandler : IMcpActionHandler
{
    private readonly ILogger<AnalyzerReplicaActionHandler> _logger;
    private readonly CodeAnalyzerService _codeAnalyzerService;
    private readonly SecurityVulnerabilityAnalyzer _securityAnalyzer;
    private readonly ImprovementSuggestionGenerator _improvementSuggestionGenerator;

    /// <summary>
    /// Initializes a new instance of the AnalyzerReplicaActionHandler class
    /// </summary>
    /// <param name="logger">Logger instance</param>
    /// <param name="codeAnalyzerService">Code analyzer service</param>
    /// <param name="securityAnalyzer">Security vulnerability analyzer</param>
    /// <param name="improvementSuggestionGenerator">Improvement suggestion generator</param>
    public AnalyzerReplicaActionHandler(
        ILogger<AnalyzerReplicaActionHandler> logger,
        CodeAnalyzerService codeAnalyzerService,
        SecurityVulnerabilityAnalyzer securityAnalyzer,
        ImprovementSuggestionGenerator improvementSuggestionGenerator)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _codeAnalyzerService = codeAnalyzerService ?? throw new ArgumentNullException(nameof(codeAnalyzerService));
        _securityAnalyzer = securityAnalyzer ?? throw new ArgumentNullException(nameof(securityAnalyzer));
        _improvementSuggestionGenerator = improvementSuggestionGenerator ?? throw new ArgumentNullException(nameof(improvementSuggestionGenerator));
    }

    /// <inheritdoc/>
    public string ActionType => "analyze_code";

    /// <inheritdoc/>
    public async Task<McpActionResult> HandleActionAsync(McpAction action)
    {
        _logger.LogInformation("Handling analyze_code action");

        try
        {
            // Extract parameters from the action
            var parameters = action.Parameters;
            var filePath = parameters.TryGetProperty("file_path", out var filePathElement)
                ? filePathElement.GetString()
                : null;

            var fileContent = parameters.TryGetProperty("file_content", out var fileContentElement)
                ? fileContentElement.GetString()
                : null;

            // Validate parameters
            if (string.IsNullOrEmpty(filePath))
            {
                return McpActionResult.CreateFailure("File path is required", action.Id);
            }

            // If file content is provided, save it to a temporary file
            var tempFilePath = filePath;
            if (!string.IsNullOrEmpty(fileContent))
            {
                tempFilePath = Path.Combine(Path.GetTempPath(), Path.GetFileName(filePath));
                await File.WriteAllTextAsync(tempFilePath, fileContent);
            }

            // Analyze the file
            var analysisResult = await _codeAnalyzerService.AnalyzeFileAsync(tempFilePath);
            if (analysisResult == null)
            {
                return McpActionResult.CreateFailure("Failed to analyze file", action.Id);
            }

            // Generate improvement suggestions
            var improvementSuggestion = await _improvementSuggestionGenerator.GenerateSuggestionAsync(analysisResult);

            // Clean up temporary file
            if (tempFilePath != filePath && File.Exists(tempFilePath))
            {
                File.Delete(tempFilePath);
            }

            // Convert the analysis result to a JSON-friendly format
            var resultObj = new
            {
                file_path = analysisResult.FilePath,
                needs_improvement = analysisResult.NeedsImprovement,
                issues = ConvertIssues(analysisResult.Issues),
                metrics = analysisResult.Metrics,
                summary = improvementSuggestion.Summary,
                detailed_description = improvementSuggestion.DetailedDescription,
                improved_content = improvementSuggestion.ImprovedContent
            };

            var resultJson = JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(resultObj));
            return McpActionResult.CreateSuccess(resultJson, action.Id);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error handling analyze_code action");
            return McpActionResult.CreateFailure(ex.Message, action.Id);
        }
    }

    /// <summary>
    /// Converts code issues to a JSON-friendly format
    /// </summary>
    /// <param name="issues">List of code issues</param>
    /// <returns>List of JSON-friendly issues</returns>
    private List<object> ConvertIssues(List<CodeAnalysis.CodeIssue> issues)
    {
        var result = new List<object>();
        foreach (var issue in issues)
        {
            result.Add(new
            {
                type = issue.Type.ToString().ToLowerInvariant(),
                severity = issue.Severity.ToString().ToLowerInvariant(),
                description = issue.Description,
                line_number = issue.LineNumber,
                column_number = issue.ColumnNumber,
                length = issue.Length,
                code_segment = issue.CodeSegment,
                suggested_fix = issue.SuggestedFix
            });
        }
        return result;
    }
}