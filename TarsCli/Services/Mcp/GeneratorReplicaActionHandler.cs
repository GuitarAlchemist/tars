using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsCli.Services.CodeAnalysis;
using TarsCli.Services.CodeGeneration;

namespace TarsCli.Services.Mcp
{
    /// <summary>
    /// Action handler for generator replica
    /// </summary>
    public class GeneratorReplicaActionHandler : IMcpActionHandler
    {
        private readonly ILogger<GeneratorReplicaActionHandler> _logger;
        private readonly CodeAnalyzerService _codeAnalyzerService;
        private readonly CodeGeneratorService _codeGeneratorService;

        /// <summary>
        /// Initializes a new instance of the GeneratorReplicaActionHandler class
        /// </summary>
        /// <param name="logger">Logger instance</param>
        /// <param name="codeAnalyzerService">Code analyzer service</param>
        /// <param name="codeGeneratorService">Code generator service</param>
        public GeneratorReplicaActionHandler(
            ILogger<GeneratorReplicaActionHandler> logger,
            CodeAnalyzerService codeAnalyzerService,
            CodeGeneratorService codeGeneratorService)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _codeAnalyzerService = codeAnalyzerService ?? throw new ArgumentNullException(nameof(codeAnalyzerService));
            _codeGeneratorService = codeGeneratorService ?? throw new ArgumentNullException(nameof(codeGeneratorService));
        }

        /// <inheritdoc/>
        public string ActionName => "generate_code";

        /// <inheritdoc/>
        public async Task<JsonElement> HandleActionAsync(JsonElement request)
        {
            _logger.LogInformation("Handling generate_code action");

            try
            {
                // Extract parameters from the request
                var filePath = request.TryGetProperty("file_path", out var filePathElement)
                    ? filePathElement.GetString()
                    : null;

                var originalContent = request.TryGetProperty("original_content", out var originalContentElement)
                    ? originalContentElement.GetString()
                    : null;

                var analysisResultJson = request.TryGetProperty("analysis_result", out var analysisResultElement)
                    ? analysisResultElement
                    : default;

                // Validate parameters
                if (string.IsNullOrEmpty(filePath))
                {
                    return JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(new
                    {
                        success = false,
                        error = "File path is required"
                    }));
                }

                // If original content is provided, save it to a temporary file
                var tempFilePath = filePath;
                if (!string.IsNullOrEmpty(originalContent))
                {
                    tempFilePath = Path.Combine(Path.GetTempPath(), Path.GetFileName(filePath));
                    await File.WriteAllTextAsync(tempFilePath, originalContent);
                }

                // Analyze the file if analysis result is not provided
                CodeAnalysisResult analysisResult = null;
                if (analysisResultJson.ValueKind != JsonValueKind.Undefined)
                {
                    // Convert the JSON analysis result to a CodeAnalysisResult object
                    analysisResult = ConvertFromJson(analysisResultJson, tempFilePath);
                }
                else
                {
                    // Analyze the file
                    analysisResult = await _codeAnalyzerService.AnalyzeFileAsync(tempFilePath);
                }

                if (analysisResult == null)
                {
                    return JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(new
                    {
                        success = false,
                        error = "Failed to analyze file"
                    }));
                }

                // Generate improved code
                var generationResult = await _codeGeneratorService.GenerateCodeAsync(tempFilePath);
                if (generationResult == null)
                {
                    return JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(new
                    {
                        success = false,
                        error = "Failed to generate code"
                    }));
                }

                // Clean up temporary file
                if (tempFilePath != filePath && File.Exists(tempFilePath))
                {
                    File.Delete(tempFilePath);
                }

                // Convert the generation result to a JSON-friendly format
                var result = new
                {
                    success = true,
                    file_path = generationResult.FilePath,
                    original_content = generationResult.OriginalContent,
                    generated_content = generationResult.GeneratedContent,
                    changes = ConvertChanges(generationResult.Changes)
                };

                return JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(result));
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error handling generate_code action");
                return JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(new
                {
                    success = false,
                    error = ex.Message
                }));
            }
        }

        /// <summary>
        /// Converts code changes to a JSON-friendly format
        /// </summary>
        /// <param name="changes">List of code changes</param>
        /// <returns>List of JSON-friendly changes</returns>
        private List<object> ConvertChanges(List<CodeChange> changes)
        {
            var result = new List<object>();
            foreach (var change in changes)
            {
                result.Add(new
                {
                    type = change.Type.ToString().ToLowerInvariant(),
                    description = change.Description,
                    line_number = change.LineNumber,
                    original_code = change.OriginalCode,
                    new_code = change.NewCode
                });
            }
            return result;
        }

        /// <summary>
        /// Converts a JSON analysis result to a CodeAnalysisResult object
        /// </summary>
        /// <param name="json">JSON analysis result</param>
        /// <param name="filePath">File path</param>
        /// <returns>CodeAnalysisResult object</returns>
        private CodeAnalysisResult ConvertFromJson(JsonElement json, string filePath)
        {
            var result = new CodeAnalysisResult
            {
                FilePath = filePath
            };

            // Extract needs_improvement
            if (json.TryGetProperty("needs_improvement", out var needsImprovementElement))
            {
                result.NeedsImprovement = needsImprovementElement.GetBoolean();
            }

            // Extract issues
            if (json.TryGetProperty("issues", out var issuesElement) && issuesElement.ValueKind == JsonValueKind.Array)
            {
                foreach (var issueElement in issuesElement.EnumerateArray())
                {
                    var issue = new CodeIssue();

                    // Extract type
                    if (issueElement.TryGetProperty("type", out var typeElement))
                    {
                        var typeString = typeElement.GetString();
                        if (Enum.TryParse<CodeIssueType>(typeString, true, out var type))
                        {
                            issue.Type = type;
                        }
                    }

                    // Extract severity
                    if (issueElement.TryGetProperty("severity", out var severityElement))
                    {
                        var severityString = severityElement.GetString();
                        if (Enum.TryParse<IssueSeverity>(severityString, true, out var severity))
                        {
                            issue.Severity = severity;
                        }
                    }

                    // Extract description
                    if (issueElement.TryGetProperty("description", out var descriptionElement))
                    {
                        issue.Description = descriptionElement.GetString();
                    }

                    // Extract line_number
                    if (issueElement.TryGetProperty("line_number", out var lineNumberElement))
                    {
                        issue.LineNumber = lineNumberElement.GetInt32();
                    }

                    // Extract column_number
                    if (issueElement.TryGetProperty("column_number", out var columnNumberElement))
                    {
                        issue.ColumnNumber = columnNumberElement.GetInt32();
                    }

                    // Extract length
                    if (issueElement.TryGetProperty("length", out var lengthElement))
                    {
                        issue.Length = lengthElement.GetInt32();
                    }

                    // Extract code_segment
                    if (issueElement.TryGetProperty("code_segment", out var codeSegmentElement))
                    {
                        issue.CodeSegment = codeSegmentElement.GetString();
                    }

                    // Extract suggested_fix
                    if (issueElement.TryGetProperty("suggested_fix", out var suggestedFixElement))
                    {
                        issue.SuggestedFix = suggestedFixElement.GetString();
                    }

                    result.Issues.Add(issue);
                }
            }

            // Extract metrics
            if (json.TryGetProperty("metrics", out var metricsElement) && metricsElement.ValueKind == JsonValueKind.Object)
            {
                foreach (var metric in metricsElement.EnumerateObject())
                {
                    if (metric.Value.ValueKind == JsonValueKind.Number)
                    {
                        result.Metrics[metric.Name] = metric.Value.GetDouble();
                    }
                }
            }

            return result;
        }
    }
}
