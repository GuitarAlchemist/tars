using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.Extensions.Logging;

namespace TarsEngine.Services.CodeAnalysis
{
    /// <summary>
    /// Generates unified analysis reports.
    /// </summary>
    public class AnalysisReportGenerator
    {
        private readonly ILogger<AnalysisReportGenerator> _logger;
        private readonly PatternDetector _patternDetector;

        /// <summary>
        /// Initializes a new instance of the <see cref="AnalysisReportGenerator"/> class.
        /// </summary>
        /// <param name="logger">The logger.</param>
        /// <param name="patternDetector">The pattern detector.</param>
        public AnalysisReportGenerator(
            ILogger<AnalysisReportGenerator> logger,
            PatternDetector patternDetector)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _patternDetector = patternDetector ?? throw new ArgumentNullException(nameof(patternDetector));
        }

        /// <summary>
        /// Represents an analysis report.
        /// </summary>
        public class AnalysisReport
        {
            /// <summary>
            /// Gets or sets the file path.
            /// </summary>
            public string FilePath { get; set; }

            /// <summary>
            /// Gets or sets the analysis date.
            /// </summary>
            public DateTime AnalysisDate { get; set; }

            /// <summary>
            /// Gets or sets the detected patterns.
            /// </summary>
            public List<PatternDetector.DetectedPattern> DetectedPatterns { get; set; } = new List<PatternDetector.DetectedPattern>();

            /// <summary>
            /// Gets or sets the summary.
            /// </summary>
            public string Summary { get; set; }

            /// <summary>
            /// Gets or sets the recommendations.
            /// </summary>
            public List<string> Recommendations { get; set; } = new List<string>();

            /// <summary>
            /// Gets or sets the errors.
            /// </summary>
            public List<string> Errors { get; set; } = new List<string>();

            /// <summary>
            /// Gets or sets a value indicating whether the analysis was successful.
            /// </summary>
            public bool Success => Errors.Count == 0;
        }

        /// <summary>
        /// Generates an analysis report for the specified code.
        /// </summary>
        /// <param name="code">The code to analyze.</param>
        /// <param name="filePath">The file path.</param>
        /// <returns>The analysis report.</returns>
        public AnalysisReport GenerateReport(string code, string filePath)
        {
            _logger.LogInformation("Generating analysis report for {FilePath}", filePath);

            var report = new AnalysisReport
            {
                FilePath = filePath,
                AnalysisDate = DateTime.Now
            };

            try
            {
                // Detect patterns
                var patterns = _patternDetector.DetectPatterns(code);
                report.DetectedPatterns = patterns;

                // Generate summary
                report.Summary = GenerateSummary(patterns);

                // Generate recommendations
                report.Recommendations = GenerateRecommendations(patterns);

                return report;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error generating analysis report");
                report.Errors.Add($"Error generating analysis report: {ex.Message}");
                return report;
            }
        }

        /// <summary>
        /// Generates a markdown report for the specified analysis report.
        /// </summary>
        /// <param name="report">The analysis report.</param>
        /// <returns>The markdown report.</returns>
        public string GenerateMarkdownReport(AnalysisReport report)
        {
            _logger.LogInformation("Generating markdown report for {FilePath}", report.FilePath);

            var markdown = new StringBuilder();

            // Add header
            markdown.AppendLine("# Code Analysis Report");
            markdown.AppendLine();
            markdown.AppendLine($"**File:** {report.FilePath}");
            markdown.AppendLine($"**Date:** {report.AnalysisDate}");
            markdown.AppendLine();

            // Add summary
            markdown.AppendLine("## Summary");
            markdown.AppendLine();
            markdown.AppendLine(report.Summary);
            markdown.AppendLine();

            // Add patterns by category
            var patternsByCategory = report.DetectedPatterns
                .GroupBy(p => p.Category)
                .OrderBy(g => g.Key);

            foreach (var category in patternsByCategory)
            {
                markdown.AppendLine($"## {category.Key} Issues");
                markdown.AppendLine();

                var patternsBySeverity = category
                    .OrderByDescending(p => GetSeverityRank(p.Severity));

                foreach (var pattern in patternsBySeverity)
                {
                    markdown.AppendLine($"### {pattern.Name} ({pattern.Severity})");
                    markdown.AppendLine();
                    markdown.AppendLine($"**Description:** {pattern.Description}");
                    markdown.AppendLine($"**Location:** {pattern.Location}");
                    markdown.AppendLine();
                    markdown.AppendLine("```csharp");
                    markdown.AppendLine(pattern.Code);
                    markdown.AppendLine("```");
                    markdown.AppendLine();
                    markdown.AppendLine($"**Suggested Fix:** {pattern.SuggestedFix}");
                    markdown.AppendLine();
                }
            }

            // Add recommendations
            markdown.AppendLine("## Recommendations");
            markdown.AppendLine();
            foreach (var recommendation in report.Recommendations)
            {
                markdown.AppendLine($"- {recommendation}");
            }
            markdown.AppendLine();

            // Add errors
            if (report.Errors.Count > 0)
            {
                markdown.AppendLine("## Errors");
                markdown.AppendLine();
                foreach (var error in report.Errors)
                {
                    markdown.AppendLine($"- {error}");
                }
                markdown.AppendLine();
            }

            return markdown.ToString();
        }

        /// <summary>
        /// Generates a JSON report for the specified analysis report.
        /// </summary>
        /// <param name="report">The analysis report.</param>
        /// <returns>The JSON report.</returns>
        public string GenerateJsonReport(AnalysisReport report)
        {
            _logger.LogInformation("Generating JSON report for {FilePath}", report.FilePath);

            // Convert the report to JSON
            var json = System.Text.Json.JsonSerializer.Serialize(report, new System.Text.Json.JsonSerializerOptions
            {
                WriteIndented = true
            });

            return json;
        }

        /// <summary>
        /// Generates a HTML report for the specified analysis report.
        /// </summary>
        /// <param name="report">The analysis report.</param>
        /// <returns>The HTML report.</returns>
        public string GenerateHtmlReport(AnalysisReport report)
        {
            _logger.LogInformation("Generating HTML report for {FilePath}", report.FilePath);

            var html = new StringBuilder();

            // Add header
            html.AppendLine("<!DOCTYPE html>");
            html.AppendLine("<html>");
            html.AppendLine("<head>");
            html.AppendLine("  <title>Code Analysis Report</title>");
            html.AppendLine("  <style>");
            html.AppendLine("    body { font-family: Arial, sans-serif; margin: 20px; }");
            html.AppendLine("    h1 { color: #333; }");
            html.AppendLine("    h2 { color: #666; margin-top: 30px; }");
            html.AppendLine("    h3 { color: #999; }");
            html.AppendLine("    .code { background-color: #f5f5f5; padding: 10px; border-radius: 5px; font-family: monospace; white-space: pre; }");
            html.AppendLine("    .critical { color: #d9534f; }");
            html.AppendLine("    .high { color: #f0ad4e; }");
            html.AppendLine("    .medium { color: #5bc0de; }");
            html.AppendLine("    .low { color: #5cb85c; }");
            html.AppendLine("  </style>");
            html.AppendLine("</head>");
            html.AppendLine("<body>");
            html.AppendLine("  <h1>Code Analysis Report</h1>");
            html.AppendLine($"  <p><strong>File:</strong> {report.FilePath}</p>");
            html.AppendLine($"  <p><strong>Date:</strong> {report.AnalysisDate}</p>");

            // Add summary
            html.AppendLine("  <h2>Summary</h2>");
            html.AppendLine($"  <p>{report.Summary}</p>");

            // Add patterns by category
            var patternsByCategory = report.DetectedPatterns
                .GroupBy(p => p.Category)
                .OrderBy(g => g.Key);

            foreach (var category in patternsByCategory)
            {
                html.AppendLine($"  <h2>{category.Key} Issues</h2>");

                var patternsBySeverity = category
                    .OrderByDescending(p => GetSeverityRank(p.Severity));

                foreach (var pattern in patternsBySeverity)
                {
                    var severityClass = pattern.Severity.ToLower();
                    html.AppendLine($"  <h3>{pattern.Name} <span class=\"{severityClass}\">({pattern.Severity})</span></h3>");
                    html.AppendLine($"  <p><strong>Description:</strong> {pattern.Description}</p>");
                    html.AppendLine($"  <p><strong>Location:</strong> {pattern.Location}</p>");
                    html.AppendLine("  <div class=\"code\">");
                    html.AppendLine(pattern.Code);
                    html.AppendLine("  </div>");
                    html.AppendLine($"  <p><strong>Suggested Fix:</strong> {pattern.SuggestedFix}</p>");
                }
            }

            // Add recommendations
            html.AppendLine("  <h2>Recommendations</h2>");
            html.AppendLine("  <ul>");
            foreach (var recommendation in report.Recommendations)
            {
                html.AppendLine($"    <li>{recommendation}</li>");
            }
            html.AppendLine("  </ul>");

            // Add errors
            if (report.Errors.Count > 0)
            {
                html.AppendLine("  <h2>Errors</h2>");
                html.AppendLine("  <ul>");
                foreach (var error in report.Errors)
                {
                    html.AppendLine($"    <li>{error}</li>");
                }
                html.AppendLine("  </ul>");
            }

            html.AppendLine("</body>");
            html.AppendLine("</html>");

            return html.ToString();
        }

        private string GenerateSummary(List<PatternDetector.DetectedPattern> patterns)
        {
            var summary = new StringBuilder();

            var totalPatterns = patterns.Count;
            var criticalPatterns = patterns.Count(p => p.Severity == "Critical");
            var highPatterns = patterns.Count(p => p.Severity == "High");
            var mediumPatterns = patterns.Count(p => p.Severity == "Medium");
            var lowPatterns = patterns.Count(p => p.Severity == "Low");

            summary.AppendLine($"Found {totalPatterns} issues in the code:");
            summary.AppendLine($"- {criticalPatterns} critical issues");
            summary.AppendLine($"- {highPatterns} high severity issues");
            summary.AppendLine($"- {mediumPatterns} medium severity issues");
            summary.AppendLine($"- {lowPatterns} low severity issues");

            var patternsByCategory = patterns
                .GroupBy(p => p.Category)
                .OrderBy(g => g.Key);

            summary.AppendLine();
            summary.AppendLine("Issues by category:");
            foreach (var category in patternsByCategory)
            {
                summary.AppendLine($"- {category.Count()} {category.Key} issues");
            }

            return summary.ToString();
        }

        private List<string> GenerateRecommendations(List<PatternDetector.DetectedPattern> patterns)
        {
            var recommendations = new List<string>();

            // Add general recommendations
            if (patterns.Any(p => p.Category == "Performance"))
            {
                recommendations.Add("Improve performance by addressing the identified performance issues.");
            }

            if (patterns.Any(p => p.Category == "Error Handling"))
            {
                recommendations.Add("Enhance error handling to make the code more robust.");
            }

            if (patterns.Any(p => p.Category == "Maintainability"))
            {
                recommendations.Add("Improve code maintainability to make it easier to understand and modify.");
            }

            if (patterns.Any(p => p.Category == "Security"))
            {
                recommendations.Add("Address security vulnerabilities to protect against potential attacks.");
            }

            // Add specific recommendations
            if (patterns.Any(p => p.Name == "String Concatenation in Loop"))
            {
                recommendations.Add("Use StringBuilder instead of string concatenation in loops for better performance.");
            }

            if (patterns.Any(p => p.Name == "LINQ in Loop"))
            {
                recommendations.Add("Move LINQ operations outside of loops or cache the results to avoid redundant computations.");
            }

            if (patterns.Any(p => p.Name == "Missing Null Check"))
            {
                recommendations.Add("Add null checks for parameters to prevent NullReferenceException.");
            }

            if (patterns.Any(p => p.Name == "Potential Division by Zero"))
            {
                recommendations.Add("Add checks to prevent division by zero.");
            }

            if (patterns.Any(p => p.Name == "Magic Number"))
            {
                recommendations.Add("Replace magic numbers with named constants for better readability.");
            }

            if (patterns.Any(p => p.Name == "Long Method"))
            {
                recommendations.Add("Break down long methods into smaller, more focused methods.");
            }

            if (patterns.Any(p => p.Name == "SQL Injection Vulnerability"))
            {
                recommendations.Add("Use parameterized queries or prepared statements instead of string concatenation to prevent SQL injection.");
            }

            if (patterns.Any(p => p.Name == "Hardcoded Credentials"))
            {
                recommendations.Add("Store credentials in a secure configuration or use a secret management system.");
            }

            return recommendations;
        }

        private int GetSeverityRank(string severity)
        {
            switch (severity)
            {
                case "Critical":
                    return 4;
                case "High":
                    return 3;
                case "Medium":
                    return 2;
                case "Low":
                    return 1;
                default:
                    return 0;
            }
        }
    }
}
