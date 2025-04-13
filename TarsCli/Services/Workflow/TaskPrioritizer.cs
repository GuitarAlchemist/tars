using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Extensions.Logging;
using TarsCli.Services.CodeAnalysis;

namespace TarsCli.Services.Workflow
{
    /// <summary>
    /// Service for prioritizing tasks
    /// </summary>
    public class TaskPrioritizer
    {
        private readonly ILogger<TaskPrioritizer> _logger;

        /// <summary>
        /// Initializes a new instance of the TaskPrioritizer class
        /// </summary>
        /// <param name="logger">Logger instance</param>
        public TaskPrioritizer(ILogger<TaskPrioritizer> logger)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        /// <summary>
        /// Prioritizes files for improvement
        /// </summary>
        /// <param name="analysisResults">Analysis results</param>
        /// <param name="maxFiles">Maximum number of files to prioritize</param>
        /// <returns>Prioritized list of files</returns>
        public List<string> PrioritizeFiles(List<CodeAnalysisResult> analysisResults, int maxFiles = 10)
        {
            _logger.LogInformation($"Prioritizing {analysisResults.Count} files for improvement");

            try
            {
                // Filter files that need improvement
                var filesToImprove = analysisResults.Where(r => r.NeedsImprovement).ToList();

                // Calculate priority scores
                var priorityScores = new Dictionary<string, double>();
                foreach (var result in filesToImprove)
                {
                    var score = CalculateFilePriorityScore(result);
                    priorityScores[result.FilePath] = score;
                }

                // Sort files by priority score (descending)
                var prioritizedFiles = priorityScores
                    .OrderByDescending(kv => kv.Value)
                    .Take(maxFiles)
                    .Select(kv => kv.Key)
                    .ToList();

                _logger.LogInformation($"Prioritized {prioritizedFiles.Count} files for improvement");
                return prioritizedFiles;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error prioritizing files for improvement");
                return analysisResults.Take(maxFiles).Select(r => r.FilePath).ToList();
            }
        }

        /// <summary>
        /// Prioritizes issues for fixing
        /// </summary>
        /// <param name="issues">List of issues</param>
        /// <param name="maxIssues">Maximum number of issues to prioritize</param>
        /// <returns>Prioritized list of issues</returns>
        public List<CodeIssue> PrioritizeIssues(List<CodeIssue> issues, int maxIssues = 100)
        {
            _logger.LogInformation($"Prioritizing {issues.Count} issues for fixing");

            try
            {
                // Calculate priority scores
                var priorityScores = new Dictionary<CodeIssue, double>();
                foreach (var issue in issues)
                {
                    var score = CalculateIssuePriorityScore(issue);
                    priorityScores[issue] = score;
                }

                // Sort issues by priority score (descending)
                var prioritizedIssues = priorityScores
                    .OrderByDescending(kv => kv.Value)
                    .Take(maxIssues)
                    .Select(kv => kv.Key)
                    .ToList();

                _logger.LogInformation($"Prioritized {prioritizedIssues.Count} issues for fixing");
                return prioritizedIssues;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error prioritizing issues for fixing");
                return issues.Take(maxIssues).ToList();
            }
        }

        /// <summary>
        /// Calculates a priority score for a file
        /// </summary>
        /// <param name="analysisResult">Analysis result</param>
        /// <returns>Priority score</returns>
        private double CalculateFilePriorityScore(CodeAnalysisResult analysisResult)
        {
            // Base score
            double score = 0;

            // Add points for each issue based on severity
            foreach (var issue in analysisResult.Issues)
            {
                switch (issue.Severity)
                {
                    case IssueSeverity.Critical:
                        score += 10;
                        break;
                    case IssueSeverity.Error:
                        score += 5;
                        break;
                    case IssueSeverity.Warning:
                        score += 2;
                        break;
                    case IssueSeverity.Info:
                        score += 1;
                        break;
                }
            }

            // Add points for issue types
            var securityIssues = analysisResult.Issues.Count(i => i.Type == CodeIssueType.Security);
            var performanceIssues = analysisResult.Issues.Count(i => i.Type == CodeIssueType.Performance);
            var reliabilityIssues = analysisResult.Issues.Count(i => i.Type == CodeIssueType.Reliability);

            score += securityIssues * 5; // Security issues are high priority
            score += performanceIssues * 3; // Performance issues are medium priority
            score += reliabilityIssues * 4; // Reliability issues are medium-high priority

            // Adjust score based on file metrics
            if (analysisResult.Metrics.TryGetValue("LineCount", out var lineCount))
            {
                // Larger files get a slight boost (more impact)
                score *= 1 + Math.Log10(lineCount) / 10;
            }

            if (analysisResult.Metrics.TryGetValue("Complexity", out var complexity))
            {
                // More complex files get a boost (more benefit from improvement)
                score *= 1 + complexity / 100;
            }

            return score;
        }

        /// <summary>
        /// Calculates a priority score for an issue
        /// </summary>
        /// <param name="issue">Code issue</param>
        /// <returns>Priority score</returns>
        private double CalculateIssuePriorityScore(CodeIssue issue)
        {
            // Base score based on severity
            double score = 0;
            switch (issue.Severity)
            {
                case IssueSeverity.Critical:
                    score = 100;
                    break;
                case IssueSeverity.Error:
                    score = 50;
                    break;
                case IssueSeverity.Warning:
                    score = 20;
                    break;
                case IssueSeverity.Info:
                    score = 10;
                    break;
            }

            // Adjust score based on issue type
            switch (issue.Type)
            {
                case CodeIssueType.Security:
                    score *= 2.0; // Security issues are highest priority
                    break;
                case CodeIssueType.Reliability:
                    score *= 1.5; // Reliability issues are high priority
                    break;
                case CodeIssueType.Performance:
                    score *= 1.3; // Performance issues are medium-high priority
                    break;
                case CodeIssueType.Functional:
                    score *= 1.2; // Functional issues are medium priority
                    break;
                case CodeIssueType.Maintainability:
                    score *= 1.1; // Maintainability issues are medium-low priority
                    break;
                case CodeIssueType.Documentation:
                    score *= 0.8; // Documentation issues are low priority
                    break;
                case CodeIssueType.Style:
                    score *= 0.7; // Style issues are lowest priority
                    break;
            }

            // Adjust score based on whether a fix is suggested
            if (!string.IsNullOrEmpty(issue.SuggestedFix))
            {
                score *= 1.2; // Issues with suggested fixes are easier to address
            }

            return score;
        }
    }
}
