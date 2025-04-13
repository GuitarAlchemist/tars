using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsCli.Services.CodeAnalysis;

namespace TarsCli.Services.Mcp
{
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
        public string ActionName => "analyze_code";

        /// <inheritdoc/>
        public async Task<JsonElement> HandleActionAsync(JsonElement request)
        {
            _logger.LogInformation("Handling analyze_code action");

            try
            {
                // Extract parameters from the request
                var filePath = request.TryGetProperty("file_path", out var filePathElement)
                    ? filePathElement.GetString()
                    : null;

                var fileContent = request.TryGetProperty("file_content", out var fileContentElement)
                    ? fileContentElement.GetString()
                    : null;

                // Validate parameters
                if (string.IsNullOrEmpty(filePath))
                {
                    return JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(new
                    {
                        success = false,
                        error = "File path is required"
                    }));
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
                    return JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(new
                    {
                        success = false,
                        error = "Failed to analyze file"
                    }));
                }

                // Generate improvement suggestions
                var improvementSuggestion = await _improvementSuggestionGenerator.GenerateSuggestionAsync(analysisResult);

                // Clean up temporary file
                if (tempFilePath != filePath && File.Exists(tempFilePath))
                {
                    File.Delete(tempFilePath);
                }

                // Convert the analysis result to a JSON-friendly format
                var result = new
                {
                    success = true,
                    file_path = analysisResult.FilePath,
                    needs_improvement = analysisResult.NeedsImprovement,
                    issues = ConvertIssues(analysisResult.Issues),
                    metrics = analysisResult.Metrics,
                    summary = improvementSuggestion.Summary,
                    detailed_description = improvementSuggestion.DetailedDescription,
                    improved_content = improvementSuggestion.ImprovedContent
                };

                return JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(result));
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error handling analyze_code action");
                return JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(new
                {
                    success = false,
                    error = ex.Message
                }));
            }
        }

        /// <summary>
        /// Converts code issues to a JSON-friendly format
        /// </summary>
        /// <param name="issues">List of code issues</param>
        /// <returns>List of JSON-friendly issues</returns>
        private List<object> ConvertIssues(List<CodeIssue> issues)
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
}
