namespace TarsEngine.Services.Abstractions.Models.CodeAnalysis
{
    /// <summary>
    /// Represents the result of a code analysis operation.
    /// </summary>
    public class CodeAnalysisResult
    {
        /// <summary>
        /// Gets or sets the list of issues found during analysis.
        /// </summary>
        public List<CodeIssue> Issues { get; set; } = new List<CodeIssue>();

        /// <summary>
        /// Gets or sets the list of metrics calculated during analysis.
        /// </summary>
        public List<CodeMetric> Metrics { get; set; } = new List<CodeMetric>();

        /// <summary>
        /// Gets or sets the file path that was analyzed.
        /// </summary>
        public string? FilePath { get; set; }

        /// <summary>
        /// Gets or sets the programming language of the analyzed code.
        /// </summary>
        public string? Language { get; set; }

        /// <summary>
        /// Gets or sets the timestamp when the analysis was performed.
        /// </summary>
        public DateTime AnalysisTimestamp { get; set; } = DateTime.UtcNow;

        /// <summary>
        /// Gets a value indicating whether any issues were found.
        /// </summary>
        public bool HasIssues => Issues.Count > 0;

        /// <summary>
        /// Gets the count of issues by severity.
        /// </summary>
        /// <param name="severity">The severity level to count.</param>
        /// <returns>The number of issues with the specified severity.</returns>
        public int GetIssueCountBySeverity(IssueSeverity severity)
        {
            return Issues.Count(i => i.Severity == severity);
        }
    }
}
