using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace TarsCli.Services.Workflow
{
    /// <summary>
    /// Service for tracking workflow progress
    /// </summary>
    public class ProgressTracker
    {
        private readonly ILogger<ProgressTracker> _logger;
        private readonly string _progressDirectory;

        /// <summary>
        /// Initializes a new instance of the ProgressTracker class
        /// </summary>
        /// <param name="logger">Logger instance</param>
        public ProgressTracker(ILogger<ProgressTracker> logger)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _progressDirectory = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "progress");
            
            // Create the progress directory if it doesn't exist
            if (!Directory.Exists(_progressDirectory))
            {
                Directory.CreateDirectory(_progressDirectory);
            }
        }

        /// <summary>
        /// Records workflow progress
        /// </summary>
        /// <param name="workflow">Workflow instance</param>
        /// <returns>Task representing the asynchronous operation</returns>
        public async Task RecordProgressAsync(WorkflowInstance workflow)
        {
            _logger.LogInformation($"Recording progress for workflow {workflow.Id}");

            try
            {
                // Create a progress record
                var progressRecord = new WorkflowProgressRecord
                {
                    WorkflowId = workflow.Id,
                    WorkflowType = workflow.Type,
                    CurrentState = workflow.CurrentState,
                    Status = workflow.Status,
                    StartTime = workflow.StartTime,
                    EndTime = workflow.EndTime,
                    StateTransitions = workflow.History.Count,
                    Timestamp = DateTime.UtcNow
                };

                // Add metrics
                progressRecord.Metrics["Duration"] = (workflow.EndTime ?? DateTime.UtcNow) - workflow.StartTime;
                progressRecord.Metrics["StateTransitionsPerMinute"] = workflow.History.Count / ((workflow.EndTime ?? DateTime.UtcNow) - workflow.StartTime).TotalMinutes;

                // Add results
                foreach (var result in workflow.Results)
                {
                    progressRecord.Results[result.Key] = result.Value;
                }

                // Save the progress record
                var filePath = Path.Combine(_progressDirectory, $"{workflow.Id}_{DateTime.UtcNow:yyyyMMddHHmmss}.json");
                var json = JsonSerializer.Serialize(progressRecord, new JsonSerializerOptions { WriteIndented = true });
                await File.WriteAllTextAsync(filePath, json);

                _logger.LogInformation($"Recorded progress for workflow {workflow.Id} to {filePath}");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error recording progress for workflow {workflow.Id}");
            }
        }

        /// <summary>
        /// Gets workflow progress history
        /// </summary>
        /// <param name="workflowId">ID of the workflow</param>
        /// <returns>List of progress records</returns>
        public async Task<List<WorkflowProgressRecord>> GetProgressHistoryAsync(string workflowId)
        {
            _logger.LogInformation($"Getting progress history for workflow {workflowId}");

            try
            {
                // Find all progress records for the workflow
                var files = Directory.GetFiles(_progressDirectory, $"{workflowId}_*.json");
                var progressRecords = new List<WorkflowProgressRecord>();

                foreach (var file in files)
                {
                    var json = await File.ReadAllTextAsync(file);
                    var progressRecord = JsonSerializer.Deserialize<WorkflowProgressRecord>(json);
                    if (progressRecord != null)
                    {
                        progressRecords.Add(progressRecord);
                    }
                }

                // Sort by timestamp
                progressRecords = progressRecords.OrderBy(r => r.Timestamp).ToList();

                _logger.LogInformation($"Found {progressRecords.Count} progress records for workflow {workflowId}");
                return progressRecords;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error getting progress history for workflow {workflowId}");
                return new List<WorkflowProgressRecord>();
            }
        }

        /// <summary>
        /// Generates a progress report
        /// </summary>
        /// <param name="workflowId">ID of the workflow</param>
        /// <returns>Progress report</returns>
        public async Task<string> GenerateProgressReportAsync(string workflowId)
        {
            _logger.LogInformation($"Generating progress report for workflow {workflowId}");

            try
            {
                // Get the progress history
                var progressHistory = await GetProgressHistoryAsync(workflowId);
                if (progressHistory.Count == 0)
                {
                    return $"No progress records found for workflow {workflowId}";
                }

                // Generate the report
                var sb = new StringBuilder();
                sb.AppendLine($"# Progress Report for Workflow {workflowId}");
                sb.AppendLine();
                sb.AppendLine($"- **Workflow Type**: {progressHistory[0].WorkflowType}");
                sb.AppendLine($"- **Start Time**: {progressHistory[0].StartTime}");
                sb.AppendLine($"- **End Time**: {progressHistory[^1].EndTime ?? "N/A"}");
                sb.AppendLine($"- **Status**: {progressHistory[^1].Status}");
                sb.AppendLine($"- **Current State**: {progressHistory[^1].CurrentState}");
                sb.AppendLine($"- **State Transitions**: {progressHistory[^1].StateTransitions}");
                sb.AppendLine();

                // Add state transition history
                sb.AppendLine("## State Transition History");
                sb.AppendLine();
                foreach (var record in progressHistory)
                {
                    sb.AppendLine($"- **{record.Timestamp}**: {record.CurrentState} ({record.Status})");
                }
                sb.AppendLine();

                // Add metrics
                sb.AppendLine("## Metrics");
                sb.AppendLine();
                foreach (var metric in progressHistory[^1].Metrics)
                {
                    sb.AppendLine($"- **{metric.Key}**: {metric.Value}");
                }
                sb.AppendLine();

                // Add results
                sb.AppendLine("## Results");
                sb.AppendLine();
                foreach (var result in progressHistory[^1].Results)
                {
                    sb.AppendLine($"- **{result.Key}**: {result.Value}");
                }

                return sb.ToString();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error generating progress report for workflow {workflowId}");
                return $"Error generating progress report: {ex.Message}";
            }
        }

        /// <summary>
        /// Calculates performance metrics for a workflow
        /// </summary>
        /// <param name="workflowId">ID of the workflow</param>
        /// <returns>Dictionary of performance metrics</returns>
        public async Task<Dictionary<string, double>> CalculatePerformanceMetricsAsync(string workflowId)
        {
            _logger.LogInformation($"Calculating performance metrics for workflow {workflowId}");

            try
            {
                // Get the progress history
                var progressHistory = await GetProgressHistoryAsync(workflowId);
                if (progressHistory.Count == 0)
                {
                    return new Dictionary<string, double>();
                }

                // Calculate metrics
                var metrics = new Dictionary<string, double>();
                var latestRecord = progressHistory[^1];

                // Duration
                var duration = (latestRecord.EndTime ?? DateTime.UtcNow) - latestRecord.StartTime;
                metrics["TotalDurationSeconds"] = duration.TotalSeconds;

                // State transitions
                metrics["StateTransitions"] = latestRecord.StateTransitions;
                metrics["StateTransitionsPerMinute"] = latestRecord.StateTransitions / duration.TotalMinutes;

                // Results
                if (latestRecord.Results.TryGetValue("SelectedFiles", out var selectedFilesObj) && selectedFilesObj is List<string> selectedFiles)
                {
                    metrics["SelectedFiles"] = selectedFiles.Count;
                }

                if (latestRecord.Results.TryGetValue("AnalysisResults", out var analysisResultsObj) && analysisResultsObj is List<CodeAnalysis.CodeAnalysisResult> analysisResults)
                {
                    metrics["AnalyzedFiles"] = analysisResults.Count;
                    metrics["FilesNeedingImprovement"] = analysisResults.Count(r => r.NeedsImprovement);
                }

                if (latestRecord.Results.TryGetValue("GenerationResults", out var generationResultsObj) && generationResultsObj is List<CodeGeneration.CodeGenerationResult> generationResults)
                {
                    metrics["GeneratedFiles"] = generationResults.Count;
                    metrics["SuccessfulGenerations"] = generationResults.Count(r => r.Success);
                }

                if (latestRecord.Results.TryGetValue("AppliedChanges", out var appliedChangesObj) && appliedChangesObj is List<string> appliedChanges)
                {
                    metrics["AppliedChanges"] = appliedChanges.Count;
                }

                _logger.LogInformation($"Calculated {metrics.Count} performance metrics for workflow {workflowId}");
                return metrics;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error calculating performance metrics for workflow {workflowId}");
                return new Dictionary<string, double>();
            }
        }
    }

    /// <summary>
    /// Record of workflow progress
    /// </summary>
    public class WorkflowProgressRecord
    {
        /// <summary>
        /// ID of the workflow
        /// </summary>
        public string WorkflowId { get; set; }

        /// <summary>
        /// Type of the workflow
        /// </summary>
        public string WorkflowType { get; set; }

        /// <summary>
        /// Current state of the workflow
        /// </summary>
        public string CurrentState { get; set; }

        /// <summary>
        /// Status of the workflow
        /// </summary>
        public WorkflowStatus Status { get; set; }

        /// <summary>
        /// Start time of the workflow
        /// </summary>
        public DateTime StartTime { get; set; }

        /// <summary>
        /// End time of the workflow
        /// </summary>
        public DateTime? EndTime { get; set; }

        /// <summary>
        /// Number of state transitions
        /// </summary>
        public int StateTransitions { get; set; }

        /// <summary>
        /// Timestamp of the record
        /// </summary>
        public DateTime Timestamp { get; set; }

        /// <summary>
        /// Metrics of the workflow
        /// </summary>
        public Dictionary<string, object> Metrics { get; set; } = new Dictionary<string, object>();

        /// <summary>
        /// Results of the workflow
        /// </summary>
        public Dictionary<string, object> Results { get; set; } = new Dictionary<string, object>();
    }
}
