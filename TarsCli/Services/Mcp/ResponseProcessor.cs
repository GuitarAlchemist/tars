using System;
using System.Collections.Generic;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace TarsCli.Services.Mcp
{
    /// <summary>
    /// Utility for processing MCP responses
    /// </summary>
    public class ResponseProcessor
    {
        private readonly ILogger<ResponseProcessor> _logger;

        /// <summary>
        /// Initializes a new instance of the ResponseProcessor class
        /// </summary>
        /// <param name="logger">Logger instance</param>
        public ResponseProcessor(ILogger<ResponseProcessor> logger)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        /// <summary>
        /// Processes a response from a replica
        /// </summary>
        /// <param name="response">Response from the replica</param>
        /// <param name="action">Action that was performed</param>
        /// <returns>Processed response</returns>
        public async Task<JsonElement> ProcessResponseAsync(JsonElement response, string action)
        {
            _logger.LogInformation($"Processing response for action: {action}");

            try
            {
                // Check if the response was successful
                var success = response.TryGetProperty("success", out var successElement) && successElement.GetBoolean();
                if (!success)
                {
                    // Extract the error message
                    var error = response.TryGetProperty("error", out var errorElement)
                        ? errorElement.GetString()
                        : "Unknown error";

                    _logger.LogError($"Error in response for action {action}: {error}");
                    return response;
                }

                // Process the response based on the action
                switch (action)
                {
                    case "analyze_code":
                        return await ProcessAnalyzeCodeResponseAsync(response);

                    case "generate_code":
                        return await ProcessGenerateCodeResponseAsync(response);

                    case "test_code":
                        return await ProcessTestCodeResponseAsync(response);

                    case "coordinate_workflow":
                        return await ProcessCoordinateWorkflowResponseAsync(response);

                    default:
                        // For other actions, just return the response as is
                        return response;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error processing response for action {action}");
                return JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(new
                {
                    success = false,
                    error = ex.Message,
                    original_response = response
                }));
            }
        }

        /// <summary>
        /// Processes a response from the analyze_code action
        /// </summary>
        /// <param name="response">Response from the replica</param>
        /// <returns>Processed response</returns>
        private async Task<JsonElement> ProcessAnalyzeCodeResponseAsync(JsonElement response)
        {
            _logger.LogInformation("Processing analyze_code response");

            // Extract the analysis result
            var filePath = response.TryGetProperty("file_path", out var filePathElement)
                ? filePathElement.GetString()
                : null;

            var needsImprovement = response.TryGetProperty("needs_improvement", out var needsImprovementElement)
                ? needsImprovementElement.GetBoolean()
                : false;

            // Add a summary of the issues
            var issueCount = 0;
            var issueTypes = new Dictionary<string, int>();
            var issueSeverities = new Dictionary<string, int>();

            if (response.TryGetProperty("issues", out var issuesElement) && issuesElement.ValueKind == JsonValueKind.Array)
            {
                foreach (var issueElement in issuesElement.EnumerateArray())
                {
                    issueCount++;

                    // Count issue types
                    var issueType = issueElement.TryGetProperty("type", out var typeElement)
                        ? typeElement.GetString()
                        : "unknown";

                    if (!string.IsNullOrEmpty(issueType))
                    {
                        if (issueTypes.ContainsKey(issueType))
                        {
                            issueTypes[issueType]++;
                        }
                        else
                        {
                            issueTypes[issueType] = 1;
                        }
                    }

                    // Count issue severities
                    var issueSeverity = issueElement.TryGetProperty("severity", out var severityElement)
                        ? severityElement.GetString()
                        : "unknown";

                    if (!string.IsNullOrEmpty(issueSeverity))
                    {
                        if (issueSeverities.ContainsKey(issueSeverity))
                        {
                            issueSeverities[issueSeverity]++;
                        }
                        else
                        {
                            issueSeverities[issueSeverity] = 1;
                        }
                    }
                }
            }

            // Create a summary
            var summary = new
            {
                file_path = filePath,
                needs_improvement = needsImprovement,
                issue_count = issueCount,
                issue_types = issueTypes,
                issue_severities = issueSeverities
            };

            // Add the summary to the response
            var processedResponse = JsonSerializer.Deserialize<Dictionary<string, object>>(response.GetRawText());
            processedResponse["summary"] = summary;

            return JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(processedResponse));
        }

        /// <summary>
        /// Processes a response from the generate_code action
        /// </summary>
        /// <param name="response">Response from the replica</param>
        /// <returns>Processed response</returns>
        private async Task<JsonElement> ProcessGenerateCodeResponseAsync(JsonElement response)
        {
            _logger.LogInformation("Processing generate_code response");

            // Extract the generation result
            var filePath = response.TryGetProperty("file_path", out var filePathElement)
                ? filePathElement.GetString()
                : null;

            var originalContent = response.TryGetProperty("original_content", out var originalContentElement)
                ? originalContentElement.GetString()
                : null;

            var generatedContent = response.TryGetProperty("generated_content", out var generatedContentElement)
                ? generatedContentElement.GetString()
                : null;

            // Calculate the diff size
            int diffSize = 0;
            if (!string.IsNullOrEmpty(originalContent) && !string.IsNullOrEmpty(generatedContent))
            {
                diffSize = Math.Abs(generatedContent.Length - originalContent.Length);
            }

            // Count the changes
            var changeCount = 0;
            var changeTypes = new Dictionary<string, int>();

            if (response.TryGetProperty("changes", out var changesElement) && changesElement.ValueKind == JsonValueKind.Array)
            {
                foreach (var changeElement in changesElement.EnumerateArray())
                {
                    changeCount++;

                    // Count change types
                    var changeType = changeElement.TryGetProperty("type", out var typeElement)
                        ? typeElement.GetString()
                        : "unknown";

                    if (!string.IsNullOrEmpty(changeType))
                    {
                        if (changeTypes.ContainsKey(changeType))
                        {
                            changeTypes[changeType]++;
                        }
                        else
                        {
                            changeTypes[changeType] = 1;
                        }
                    }
                }
            }

            // Create a summary
            var summary = new
            {
                file_path = filePath,
                diff_size = diffSize,
                change_count = changeCount,
                change_types = changeTypes
            };

            // Add the summary to the response
            var processedResponse = JsonSerializer.Deserialize<Dictionary<string, object>>(response.GetRawText());
            processedResponse["summary"] = summary;

            return JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(processedResponse));
        }

        /// <summary>
        /// Processes a response from the test_code action
        /// </summary>
        /// <param name="response">Response from the replica</param>
        /// <returns>Processed response</returns>
        private async Task<JsonElement> ProcessTestCodeResponseAsync(JsonElement response)
        {
            _logger.LogInformation("Processing test_code response");

            // Extract the test result
            var filePath = response.TryGetProperty("source_file_path", out var filePathElement)
                ? filePathElement.GetString()
                : response.TryGetProperty("file_path", out filePathElement)
                    ? filePathElement.GetString()
                    : null;

            var testSuccess = response.TryGetProperty("test_success", out var testSuccessElement)
                ? testSuccessElement.GetBoolean()
                : false;

            var passedCount = response.TryGetProperty("passed_count", out var passedCountElement)
                ? passedCountElement.GetInt32()
                : 0;

            var failedCount = response.TryGetProperty("failed_count", out var failedCountElement)
                ? failedCountElement.GetInt32()
                : 0;

            var totalCount = response.TryGetProperty("total_count", out var totalCountElement)
                ? totalCountElement.GetInt32()
                : 0;

            // Calculate the success rate
            double successRate = totalCount > 0 ? (double)passedCount / totalCount * 100 : 0;

            // Create a summary
            var summary = new
            {
                file_path = filePath,
                test_success = testSuccess,
                passed_count = passedCount,
                failed_count = failedCount,
                total_count = totalCount,
                success_rate = successRate
            };

            // Add the summary to the response
            var processedResponse = JsonSerializer.Deserialize<Dictionary<string, object>>(response.GetRawText());
            processedResponse["summary"] = summary;

            return JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(processedResponse));
        }

        /// <summary>
        /// Processes a response from the coordinate_workflow action
        /// </summary>
        /// <param name="response">Response from the replica</param>
        /// <returns>Processed response</returns>
        private async Task<JsonElement> ProcessCoordinateWorkflowResponseAsync(JsonElement response)
        {
            _logger.LogInformation("Processing coordinate_workflow response");

            // Extract the workflow result
            var workflowId = response.TryGetProperty("workflow_id", out var workflowIdElement)
                ? workflowIdElement.GetString()
                : null;

            var currentState = response.TryGetProperty("current_state", out var currentStateElement)
                ? currentStateElement.GetString()
                : null;

            var nextState = response.TryGetProperty("next_state", out var nextStateElement)
                ? nextStateElement.GetString()
                : null;

            var status = response.TryGetProperty("status", out var statusElement)
                ? statusElement.GetString()
                : null;

            // Create a summary
            var summary = new
            {
                workflow_id = workflowId,
                current_state = currentState,
                next_state = nextState,
                status = status
            };

            // Add the summary to the response
            var processedResponse = JsonSerializer.Deserialize<Dictionary<string, object>>(response.GetRawText());
            processedResponse["summary"] = summary;

            return JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(processedResponse));
        }
    }
}
