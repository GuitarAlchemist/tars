using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace TarsCli.Services.CodeAnalysis
{
    /// <summary>
    /// Service for generating improvement suggestions based on code analysis results
    /// </summary>
    public class ImprovementSuggestionGenerator
    {
        private readonly ILogger<ImprovementSuggestionGenerator> _logger;

        /// <summary>
        /// Initializes a new instance of the ImprovementSuggestionGenerator class
        /// </summary>
        /// <param name="logger">Logger instance</param>
        public ImprovementSuggestionGenerator(ILogger<ImprovementSuggestionGenerator> logger)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        /// <summary>
        /// Generates improvement suggestions based on code analysis results
        /// </summary>
        /// <param name="analysisResult">Code analysis result</param>
        /// <returns>Improvement suggestion</returns>
        public async Task<ImprovementSuggestion> GenerateSuggestionAsync(CodeAnalysisResult analysisResult)
        {
            _logger.LogInformation($"Generating improvement suggestions for {analysisResult.FilePath}");

            try
            {
                // Create a new improvement suggestion
                var suggestion = new ImprovementSuggestion
                {
                    FilePath = analysisResult.FilePath,
                    OriginalContent = await File.ReadAllTextAsync(analysisResult.FilePath),
                    Issues = analysisResult.Issues,
                    Metrics = analysisResult.Metrics,
                    Metadata = analysisResult.Metadata
                };

                // Group issues by type
                var issuesByType = analysisResult.Issues.GroupBy(i => i.Type).ToDictionary(g => g.Key, g => g.ToList());

                // Generate summary
                suggestion.Summary = GenerateSummary(analysisResult, issuesByType);

                // Generate detailed description
                suggestion.DetailedDescription = GenerateDetailedDescription(analysisResult, issuesByType);

                // Generate improved content
                suggestion.ImprovedContent = await GenerateImprovedContentAsync(analysisResult);

                _logger.LogInformation($"Generated improvement suggestions for {analysisResult.FilePath}");
                return suggestion;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error generating improvement suggestions for {analysisResult.FilePath}");
                throw;
            }
        }

        /// <summary>
        /// Generates a summary of the improvement suggestions
        /// </summary>
        /// <param name="analysisResult">Code analysis result</param>
        /// <param name="issuesByType">Issues grouped by type</param>
        /// <returns>Summary of the improvement suggestions</returns>
        private string GenerateSummary(CodeAnalysisResult analysisResult, Dictionary<CodeIssueType, List<CodeIssue>> issuesByType)
        {
            var summary = new StringBuilder();
            summary.AppendLine($"# Improvement Suggestions for {Path.GetFileName(analysisResult.FilePath)}");
            summary.AppendLine();

            // Add metrics
            summary.AppendLine("## Code Metrics");
            summary.AppendLine();
            foreach (var metric in analysisResult.Metrics)
            {
                summary.AppendLine($"- {metric.Key}: {metric.Value}");
            }
            summary.AppendLine();

            // Add issue summary
            summary.AppendLine("## Issues Summary");
            summary.AppendLine();
            foreach (var issueType in issuesByType.Keys)
            {
                var issues = issuesByType[issueType];
                var criticalCount = issues.Count(i => i.Severity == IssueSeverity.Critical);
                var errorCount = issues.Count(i => i.Severity == IssueSeverity.Error);
                var warningCount = issues.Count(i => i.Severity == IssueSeverity.Warning);
                var infoCount = issues.Count(i => i.Severity == IssueSeverity.Info);

                summary.AppendLine($"- {issueType}: {issues.Count} issues");
                summary.AppendLine($"  - Critical: {criticalCount}");
                summary.AppendLine($"  - Error: {errorCount}");
                summary.AppendLine($"  - Warning: {warningCount}");
                summary.AppendLine($"  - Info: {infoCount}");
            }
            summary.AppendLine();

            // Add improvement summary
            summary.AppendLine("## Improvement Summary");
            summary.AppendLine();
            if (issuesByType.ContainsKey(CodeIssueType.Security))
            {
                summary.AppendLine("- Security vulnerabilities need to be addressed");
            }
            if (issuesByType.ContainsKey(CodeIssueType.Performance))
            {
                summary.AppendLine("- Performance issues need to be addressed");
            }
            if (issuesByType.ContainsKey(CodeIssueType.Maintainability))
            {
                summary.AppendLine("- Maintainability issues need to be addressed");
            }
            if (issuesByType.ContainsKey(CodeIssueType.Documentation))
            {
                summary.AppendLine("- Documentation needs to be improved");
            }
            if (issuesByType.ContainsKey(CodeIssueType.Complexity))
            {
                summary.AppendLine("- Code complexity needs to be reduced");
            }
            if (issuesByType.ContainsKey(CodeIssueType.Duplication))
            {
                summary.AppendLine("- Code duplication needs to be addressed");
            }
            if (issuesByType.ContainsKey(CodeIssueType.Style))
            {
                summary.AppendLine("- Code style needs to be improved");
            }

            return summary.ToString();
        }

        /// <summary>
        /// Generates a detailed description of the improvement suggestions
        /// </summary>
        /// <param name="analysisResult">Code analysis result</param>
        /// <param name="issuesByType">Issues grouped by type</param>
        /// <returns>Detailed description of the improvement suggestions</returns>
        private string GenerateDetailedDescription(CodeAnalysisResult analysisResult, Dictionary<CodeIssueType, List<CodeIssue>> issuesByType)
        {
            var description = new StringBuilder();
            description.AppendLine($"# Detailed Improvement Suggestions for {Path.GetFileName(analysisResult.FilePath)}");
            description.AppendLine();

            // Add detailed issues
            foreach (var issueType in issuesByType.Keys)
            {
                description.AppendLine($"## {issueType} Issues");
                description.AppendLine();

                var issues = issuesByType[issueType];
                foreach (var issue in issues.OrderByDescending(i => i.Severity))
                {
                    description.AppendLine($"### {issue.Severity}: {issue.Description}");
                    description.AppendLine();
                    description.AppendLine($"- Line: {issue.LineNumber}");
                    description.AppendLine($"- Column: {issue.ColumnNumber}");
                    description.AppendLine($"- Code: `{issue.CodeSegment}`");
                    description.AppendLine();
                    description.AppendLine("#### Suggested Fix");
                    description.AppendLine();
                    description.AppendLine(issue.SuggestedFix);
                    description.AppendLine();
                }
            }

            return description.ToString();
        }

        /// <summary>
        /// Generates improved content based on the analysis results
        /// </summary>
        /// <param name="analysisResult">Code analysis result</param>
        /// <returns>Improved content</returns>
        private async Task<string> GenerateImprovedContentAsync(CodeAnalysisResult analysisResult)
        {
            try
            {
                // Read the original content
                var originalContent = await File.ReadAllTextAsync(analysisResult.FilePath);
                var lines = originalContent.Split(new[] { "\r\n", "\r", "\n" }, StringSplitOptions.None);

                // Create a list of line modifications
                var lineModifications = new Dictionary<int, string>();

                // Apply suggested fixes
                foreach (var issue in analysisResult.Issues.OrderByDescending(i => i.Severity))
                {
                    // Skip issues without suggested fixes
                    if (string.IsNullOrEmpty(issue.SuggestedFix) || issue.SuggestedFix.StartsWith("//"))
                    {
                        continue;
                    }

                    // Apply the fix
                    if (issue.LineNumber > 0 && issue.LineNumber <= lines.Length)
                    {
                        lineModifications[issue.LineNumber - 1] = issue.SuggestedFix;
                    }
                }

                // Apply the modifications
                var improvedLines = new List<string>();
                for (int i = 0; i < lines.Length; i++)
                {
                    if (lineModifications.ContainsKey(i))
                    {
                        improvedLines.Add(lineModifications[i]);
                    }
                    else
                    {
                        improvedLines.Add(lines[i]);
                    }
                }

                return string.Join(Environment.NewLine, improvedLines);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error generating improved content for {analysisResult.FilePath}");
                return await File.ReadAllTextAsync(analysisResult.FilePath);
            }
        }
    }

    /// <summary>
    /// Represents an improvement suggestion
    /// </summary>
    public class ImprovementSuggestion
    {
        /// <summary>
        /// Path to the file
        /// </summary>
        public string FilePath { get; set; }

        /// <summary>
        /// Original content of the file
        /// </summary>
        public string OriginalContent { get; set; }

        /// <summary>
        /// Improved content of the file
        /// </summary>
        public string ImprovedContent { get; set; }

        /// <summary>
        /// Summary of the improvement suggestion
        /// </summary>
        public string Summary { get; set; }

        /// <summary>
        /// Detailed description of the improvement suggestion
        /// </summary>
        public string DetailedDescription { get; set; }

        /// <summary>
        /// List of issues found in the file
        /// </summary>
        public List<CodeIssue> Issues { get; set; } = new List<CodeIssue>();

        /// <summary>
        /// List of metrics calculated for the file
        /// </summary>
        public Dictionary<string, double> Metrics { get; set; } = new Dictionary<string, double>();

        /// <summary>
        /// Additional information about the file
        /// </summary>
        public Dictionary<string, object> Metadata { get; set; } = new Dictionary<string, object>();
    }
}
